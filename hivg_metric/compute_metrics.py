"""
HiVG Metrics Computation

Compute evaluation metrics from inference results.

Metrics:
- Basic: Success Rate, SVG Char Length, SVG Token Count, Path Count, Path Commands Count
- Visual Similarity (img2svg): SSIM, LPIPS, PSNR (pred_svg vs input_image)
- Semantic Alignment (text2svg): CLIP Score (text vs pred_image)
- Aesthetic Quality / Human Preference (both): PickScore, ImageReward, HPS
- Diversity (both): Sample diversity per prompt group using DINOv2/CLIP features

Usage:
    # text2svg metrics (basic + clip + preference)
    python hivg_metric/compute_metrics.py \
        --input results.jsonl \
        --task_type text2svg \
        --metrics basic,clip,preference \
        --output metrics.json

    # img2svg metrics (basic + visual similarity + preference)
    python hivg_metric/compute_metrics.py \
        --input results.jsonl \
        --task_type img2svg \
        --image_base_dir /path/to/images \
        --metrics basic,visual,preference \
        --output metrics.json

    # Diversity metrics (requires multiple samples per prompt)
    python hivg_metric/compute_metrics.py \
        --input results.jsonl \
        --metrics basic,diversity \
        --diversity_model dino \
        --output metrics.json

    # Quick test (basic only)
    python hivg_metric/compute_metrics.py \
        --input results.jsonl \
        --metrics basic \
        --max_samples 10

Environment Variables:
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
    HPS_ROOT: HPSv2 cache directory
    TORCH_HOME: PyTorch cache directory (for piqa LPIPS weights)

Model Cache Locations:
    - CLIP/PickScore: $HF_HOME/hub/
    - ImageReward: $HF_HOME/hub/
    - HPSv2: $HPS_ROOT or ~/.cache/hpsv2/
    - LPIPS (piqa): $TORCH_HOME/hub/
    - DINOv2: $HF_HOME/hub/
"""

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import fire
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================
# Aesthetic Scorer MLP (LAION Aesthetic Predictor)
# ============================================================

_AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac+logos+ava1-l14-linearMSE.pth"
)


class _AestheticMLP(torch.nn.Module):
    """MLP head for LAION aesthetic prediction (CLIP ViT-L/14 embedding -> score)."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


# ============================================================
# Metrics Result Container
# ============================================================


@dataclass
class MetricsResult:
    """Container for computed metrics."""

    # Basic stats
    total_samples: int = 0
    success_count: int = 0  # Token parsing success
    success_rate: float = 0.0  # success_count / total_samples
    render_success_count: int = 0  # SVG rendering success (subset of success_count)
    render_success_rate: float = 0.0  # render_success_count / total_samples
    avg_svg_char_length: float = 0.0
    avg_svg_token_count: float = 0.0
    avg_path_count: float = 0.0
    avg_path_commands_count: float = 0.0

    # Visual similarity (img2svg only)
    ssim_mean: Optional[float] = None
    ssim_std: Optional[float] = None
    lpips_mean: Optional[float] = None
    lpips_std: Optional[float] = None
    psnr_mean: Optional[float] = None
    psnr_std: Optional[float] = None
    clip_visual_sim_mean: Optional[float] = None
    clip_visual_sim_std: Optional[float] = None

    # Semantic alignment (text2svg only)
    clip_score_mean: Optional[float] = None
    clip_score_std: Optional[float] = None
    pick_score_mean: Optional[float] = None
    pick_score_std: Optional[float] = None

    # Aesthetic quality (both)
    image_reward_mean: Optional[float] = None
    image_reward_std: Optional[float] = None
    hps_mean: Optional[float] = None
    hps_std: Optional[float] = None
    aesthetic_score_mean: Optional[float] = None
    aesthetic_score_std: Optional[float] = None

    # Diversity (per-prompt group)
    diversity_dino_mean: Optional[float] = None
    diversity_dino_std: Optional[float] = None
    diversity_clip_mean: Optional[float] = None
    diversity_clip_std: Optional[float] = None

    # Per-sample details
    per_sample: list = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "basic": {
                "total_samples": self.total_samples,
                "success_count": self.success_count,
                "success_rate": self.success_rate,
                "render_success_count": self.render_success_count,
                "render_success_rate": self.render_success_rate,
                "avg_svg_char_length": self.avg_svg_char_length,
                "avg_svg_token_count": self.avg_svg_token_count,
                "avg_path_count": self.avg_path_count,
                "avg_path_commands_count": self.avg_path_commands_count,
            }
        }

        if self.ssim_mean is not None or self.clip_visual_sim_mean is not None:
            result["visual_similarity"] = {}
            if self.ssim_mean is not None:
                result["visual_similarity"]["ssim_mean"] = self.ssim_mean
                result["visual_similarity"]["ssim_std"] = self.ssim_std
            if self.lpips_mean is not None:
                result["visual_similarity"]["lpips_mean"] = self.lpips_mean
                result["visual_similarity"]["lpips_std"] = self.lpips_std
            if self.psnr_mean is not None:
                result["visual_similarity"]["psnr_mean"] = self.psnr_mean
                result["visual_similarity"]["psnr_std"] = self.psnr_std
            if self.clip_visual_sim_mean is not None:
                result["visual_similarity"]["clip_visual_sim_mean"] = self.clip_visual_sim_mean
                result["visual_similarity"]["clip_visual_sim_std"] = self.clip_visual_sim_std

        if self.clip_score_mean is not None:
            result["semantic_alignment"] = {
                "clip_score_mean": self.clip_score_mean,
                "clip_score_std": self.clip_score_std,
            }

        if self.pick_score_mean is not None or self.image_reward_mean is not None or self.hps_mean is not None or self.aesthetic_score_mean is not None:
            result["aesthetic_quality"] = {}
            if self.pick_score_mean is not None:
                result["aesthetic_quality"]["pick_score_mean"] = self.pick_score_mean
                result["aesthetic_quality"]["pick_score_std"] = self.pick_score_std
            if self.image_reward_mean is not None:
                result["aesthetic_quality"]["image_reward_mean"] = self.image_reward_mean
                result["aesthetic_quality"]["image_reward_std"] = self.image_reward_std
            if self.hps_mean is not None:
                result["aesthetic_quality"]["hps_mean"] = self.hps_mean
                result["aesthetic_quality"]["hps_std"] = self.hps_std
            if self.aesthetic_score_mean is not None:
                result["aesthetic_quality"]["aesthetic_score_mean"] = self.aesthetic_score_mean
                result["aesthetic_quality"]["aesthetic_score_std"] = self.aesthetic_score_std

        if self.diversity_dino_mean is not None or self.diversity_clip_mean is not None:
            result["diversity"] = {}
            if self.diversity_dino_mean is not None:
                result["diversity"]["dino_mean"] = self.diversity_dino_mean
                result["diversity"]["dino_std"] = self.diversity_dino_std
            if self.diversity_clip_mean is not None:
                result["diversity"]["clip_mean"] = self.diversity_clip_mean
                result["diversity"]["clip_std"] = self.diversity_clip_std

        return result


# ============================================================
# Utility Functions
# ============================================================


def _parse_background_color(background_color: str) -> tuple[int, int, int]:
    """Parse background color string to RGB tuple."""
    color = background_color.strip().lower()
    if color == "white":
        return (255, 255, 255)
    if color == "black":
        return (0, 0, 0)
    if re.fullmatch(r"#[0-9a-f]{6}", color):
        return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    raise ValueError(
        f"Invalid background color '{background_color}'. "
        "Use 'white', 'black', or hex like '#RRGGBB'."
    )


def _to_rgb_with_background(img: Image.Image, background_rgb: tuple[int, int, int]) -> Image.Image:
    """Convert PIL image to RGB with explicit alpha compositing."""
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*background_rgb, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _rgb_to_hex(background_rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return "#{:02x}{:02x}{:02x}".format(*background_rgb)


def _render_svg_to_pil(
    svg_code: str, size: int = 512, background_rgb: tuple[int, int, int] = (255, 255, 255)
) -> Optional[Image.Image]:
    """Render SVG code to PIL Image."""
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"),
            output_width=size,
            output_height=size,
            background_color=_rgb_to_hex(background_rgb),
        )
        img = _to_rgb_with_background(Image.open(io.BytesIO(png_data)), background_rgb=background_rgb)
        return img
    except Exception as e:
        print(f"[Warning] SVG render failed: {e}")
        return None


def _count_svg_paths(svg_code: str) -> int:
    """Count number of path elements in SVG."""
    return len(re.findall(r"<path\b", svg_code, re.IGNORECASE))


def _count_path_commands(svg_code: str) -> int:
    """Count number of path commands (M/L/C/S/Q/T/A/Z/H/V) in SVG.

    Path commands indicate SVG complexity more accurately than path element count.
    """
    # Extract all d="..." attributes from path elements
    d_attrs = re.findall(r'<path[^>]*\sd=["\']([^"\']*)["\']', svg_code, re.IGNORECASE)

    total_commands = 0
    for d in d_attrs:
        # Count path commands: M, L, C, S, Q, T, A, Z, H, V (case insensitive)
        commands = re.findall(r'[MLCSQTAZHVmlcsqtazhv]', d)
        total_commands += len(commands)

    return total_commands


_svg_tokenizer = None
_svg_tokenizer_loaded = False


def _get_svg_tokenizer():
    """Lazy load SVGTokenizer from hivg_tokenizer."""
    global _svg_tokenizer, _svg_tokenizer_loaded
    if not _svg_tokenizer_loaded:
        try:
            from hivg_tokenizer import SVGTokenizer
            _svg_tokenizer = SVGTokenizer(coord_range=234)
            print("[Metrics] SVGTokenizer loaded (hivg_tokenizer)")
        except ImportError:
            print("[Warning] hivg_tokenizer not found, using fallback token counting")
        except Exception as e:
            print(f"[Warning] SVGTokenizer load failed: {e}")
        _svg_tokenizer_loaded = True
    return _svg_tokenizer


def _count_svg_tokens(svg_code: str) -> int:
    """Count SVG tokens using hivg_tokenizer.

    Raises:
        RuntimeError: If SVGTokenizer is not available or tokenization fails.
    """
    tokenizer = _get_svg_tokenizer()
    if tokenizer is None:
        raise RuntimeError(
            "SVGTokenizer not available. Install hivg_tokenizer: pip install hivg_tokenizer"
        )

    try:
        tokens = tokenizer.tokenize(svg_code)
        return len(tokens)
    except Exception as e:
        raise RuntimeError(f"SVG tokenization failed: {e}") from e


def _extract_prompt_text(prompt: str) -> str:
    """Extract clean text prompt (remove special tokens)."""
    text = re.sub(r"\[T2ST\]|\[I2ST\]", "", prompt)
    text = re.sub(r"Generate SVG tokens from text\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Generate SVG tokens from image\.", "", text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_image_path_value(image_path: object) -> Optional[str]:
    """Normalize image path value from JSON record to plain string path."""
    if isinstance(image_path, str) and image_path:
        return image_path

    if isinstance(image_path, dict):
        path = image_path.get("path")
        if isinstance(path, str) and path:
            return path

    if isinstance(image_path, list) and image_path:
        return _normalize_image_path_value(image_path[0])

    return None


def _resolve_existing_image_path(image_path: object, image_base_dir: Optional[str] = None) -> Optional[str]:
    """Resolve absolute/relative image path to existing local file."""
    normalized_path = _normalize_image_path_value(image_path)
    if not normalized_path:
        return None

    candidates = [normalized_path]
    if image_base_dir and not os.path.isabs(normalized_path):
        candidates.insert(0, os.path.join(image_base_dir, normalized_path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _extract_image_path(prompt: str, image_base_dir: str) -> Optional[str]:
    """Extract input image path from prompt/record."""
    patterns = [
        r"<image>(.*?)</image>",
        r'([^\s<>"]+\.(?:png|jpg|jpeg|gif|webp))',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            img_path = match.group(1)
            resolved_path = _resolve_existing_image_path(img_path, image_base_dir=image_base_dir)
            if resolved_path:
                return resolved_path
    return None


def _load_image_paths_from_dataset(dataset_name: str, dataset_dir: str) -> list:
    """
    Load image paths from original dataset file.

    Supports sharegpt format with 'images' field.
    Returns full list without slicing to preserve index alignment with prompt_index.

    Raises:
        FileNotFoundError: If dataset_info.json or dataset file not found.
        ValueError: If dataset name not found or has no file_name.
    """
    # Read dataset_info.json
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json not found at {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if dataset_name not in dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_info.json")

    file_name = dataset_info[dataset_name].get("file_name")
    if not file_name:
        raise ValueError(f"No file_name for dataset '{dataset_name}'")

    data_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Read dataset and extract image paths
    image_paths = []
    multi_image_warned = False
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # sharegpt format: images field is a list
            images = record.get("images", [])
            if images and len(images) > 0:
                if len(images) > 1 and not multi_image_warned:
                    print("[Warning] Some records have multiple images, using first one only")
                    multi_image_warned = True
                image_paths.append(images[0])
            else:
                image_paths.append(None)

    print(f"[Metrics] Loaded {len(image_paths)} image paths from {dataset_name}")
    return image_paths


def _load_caption_lookup(caption_dataset: str) -> dict:
    """
    Load caption lookup table from HuggingFace Arrow dataset.

    Builds {md5: long_caption} dict for img2svg caption retrieval.

    Args:
        caption_dataset: Path to Arrow dataset (loaded via datasets.load_from_disk).

    Returns:
        Dict mapping md5 -> long_caption string.
    """
    from datasets import load_from_disk

    ds = load_from_disk(caption_dataset)
    print(f"[Metrics] Caption dataset has {len(ds)} rows")
    if len(ds) > 500_000:
        print(f"[Warning] Large caption dataset ({len(ds)} rows) — will consume significant memory")

    lookup = {}
    for row in ds:
        md5 = row.get("md5", "")
        caption = row.get("long_caption", "")
        if md5 and caption:
            lookup[md5] = caption

    print(f"[Metrics] Loaded {len(lookup)} captions (with valid md5+caption)")
    return lookup


def _get_filename_stem(image_path: str) -> str:
    """Extract filename stem (without extension) as lookup key."""
    return os.path.splitext(os.path.basename(image_path))[0]


def _pil_to_tensor(img: Image.Image, size: int = 512) -> torch.Tensor:
    """Convert PIL Image to tensor [1, 3, H, W] in [0, 1] range."""
    img = img.resize((size, size), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


# ============================================================
# Metrics Computer Class
# ============================================================


class MetricsComputer:
    """Compute various metrics for SVG generation."""

    def __init__(self, device: str = "cuda", render_size: int = 512, lpips_network: str = "vgg", local_files_only: bool = False):
        self.device = device
        self.render_size = render_size
        self.lpips_network = lpips_network
        self.local_files_only = local_files_only

        # Lazy-loaded models (None = not loaded, False = failed to load)
        self._ssim_fn = None
        self._lpips_fn = None
        self._psnr_fn = None
        self._clip_model = None
        self._clip_processor = None
        self._pick_model = None
        self._pick_processor = None
        self._image_reward_model = None
        self._hps_module = None
        self._aesthetic_mlp = None
        self._dino_model = None
        self._dino_processor = None

    # ==================== Image Quality Metrics (piqa) ====================

    def _get_ssim_fn(self):
        """Lazy load SSIM from piqa."""
        if self._ssim_fn is None:
            try:
                from piqa import SSIM

                self._ssim_fn = SSIM().to(self.device)
                print("[Metrics] SSIM loaded (piqa)")
            except ImportError:
                print("[Warning] piqa not found, SSIM will be skipped")
                print("  Install: pip install piqa")
        return self._ssim_fn

    def _get_lpips_fn(self):
        """Lazy load LPIPS from piqa."""
        if self._lpips_fn is None:
            try:
                from piqa import LPIPS

                self._lpips_fn = LPIPS(network=self.lpips_network).to(self.device)
                print(f"[Metrics] LPIPS loaded (piqa, {self.lpips_network} network)")
            except ImportError:
                print("[Warning] piqa not found, LPIPS will be skipped")
                print("  Install: pip install piqa")
        return self._lpips_fn

    def _get_psnr_fn(self):
        """Lazy load PSNR from piqa."""
        if self._psnr_fn is None:
            try:
                from piqa import PSNR

                self._psnr_fn = PSNR()
                print("[Metrics] PSNR loaded (piqa)")
            except ImportError:
                print("[Warning] piqa not found, PSNR will be skipped")
                print("  Install: pip install piqa")
        return self._psnr_fn

    def compute_image_metrics(self, pred_img: Image.Image, gt_img: Image.Image) -> dict:
        """Compute SSIM, LPIPS, PSNR between two images using piqa."""
        results = {}

        pred_tensor = _pil_to_tensor(pred_img, self.render_size).to(self.device)
        gt_tensor = _pil_to_tensor(gt_img, self.render_size).to(self.device)

        with torch.no_grad():
            ssim_fn = self._get_ssim_fn()
            if ssim_fn is not None:
                results["ssim"] = ssim_fn(pred_tensor, gt_tensor).item()

            lpips_fn = self._get_lpips_fn()
            if lpips_fn is not None:
                results["lpips"] = lpips_fn(pred_tensor, gt_tensor).item()

            psnr_fn = self._get_psnr_fn()
            if psnr_fn is not None:
                results["psnr"] = psnr_fn(pred_tensor, gt_tensor).item()

        return results

    # ==================== CLIP Visual Similarity (img2svg) ====================

    def compute_clip_visual_similarity(
        self, pred_img: Image.Image, gt_img: Image.Image, model_name: str = "openai/clip-vit-large-patch14"
    ) -> Optional[float]:
        """Compute CLIP visual similarity (cosine sim of image embeddings) between pred and gt."""
        model, processor = self._get_clip_model(model_name)
        if model is None:
            return None

        try:
            inputs = processor(images=[pred_img, gt_img], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                sim = (features[0] @ features[1]).item()

            return sim
        except Exception as e:
            if not hasattr(self, "_clip_vis_error_printed"):
                print(f"[Warning] CLIP visual similarity failed: {e}")
                self._clip_vis_error_printed = True
            return None

    # ==================== CLIP Score (transformers) ====================

    def _get_clip_model(self, model_name: str = "openai/clip-vit-large-patch14"):
        """
        Lazy load CLIP model from transformers.

        Available models:
        - openai/clip-vit-base-patch32
        - openai/clip-vit-base-patch16
        - openai/clip-vit-large-patch14
        - openai/clip-vit-large-patch14-336
        - laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        """
        if self._clip_model is None:
            try:
                from transformers import AutoModel, AutoProcessor

                self._clip_processor = AutoProcessor.from_pretrained(model_name, local_files_only=self.local_files_only)
                self._clip_model = AutoModel.from_pretrained(model_name, local_files_only=self.local_files_only).eval().to(self.device)
                print(f"[Metrics] CLIP loaded ({model_name})")
            except ImportError:
                print("[Warning] transformers not found, CLIP score will be skipped")
                print("  Install: pip install transformers")
            except Exception as e:
                print(f"[Warning] CLIP load failed: {e}")
        return self._clip_model, self._clip_processor

    def compute_clip_score(
        self, image: Image.Image, text: str, model_name: str = "openai/clip-vit-large-patch14"
    ) -> Optional[float]:
        """Compute CLIP score between image and text using transformers."""
        model, processor = self._get_clip_model(model_name)
        if model is None:
            return None

        try:
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # logits_per_image is [1, 1] for single image-text pair
                score = outputs.logits_per_image[0, 0].item()
                # Normalize to [0, 1] range (logits are typically in [-100, 100])
                score = score / 100.0

            return score
        except Exception as e:
            print(f"[Warning] CLIP score failed: {e}")
            return None

    # ==================== PickScore (transformers) ====================

    def _get_pick_model(self):
        """
        Lazy load PickScore model.

        Reference: https://github.com/yuvalkirstain/PickScore
        Model: yuvalkirstain/PickScore_v1
        Processor: laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        """
        if self._pick_model is None:
            try:
                from transformers import AutoModel, AutoProcessor

                self._pick_processor = AutoProcessor.from_pretrained(
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_files_only=self.local_files_only
                )
                self._pick_model = AutoModel.from_pretrained(
                    "yuvalkirstain/PickScore_v1", local_files_only=self.local_files_only
                ).eval().to(self.device)
                print("[Metrics] PickScore loaded (yuvalkirstain/PickScore_v1)")
            except ImportError:
                print("[Warning] transformers not found, PickScore will be skipped")
                print("  Install: pip install transformers")
            except Exception as e:
                print(f"[Warning] PickScore load failed: {e}")
        return self._pick_model, self._pick_processor

    def compute_pick_score(self, image: Image.Image, text: str) -> Optional[float]:
        """
        Compute PickScore between image and text.

        Higher is better (trained on human preferences).
        """
        model, processor = self._get_pick_model()
        if model is None:
            return None

        try:
            image_inputs = processor(
                images=image,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            text_inputs = processor(
                text=text,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                image_embs = model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                # Score = logit_scale * (image_embs @ text_embs.T)
                score = model.logit_scale.exp() * (image_embs @ text_embs.T)
                score = score[0, 0].item()

            return score
        except Exception as e:
            print(f"[Warning] PickScore failed: {e}")
            return None

    # ==================== ImageReward ====================

    def _get_image_reward_model(self):
        """
        Lazy load ImageReward model.

        Reference: https://github.com/THUDM/ImageReward
        """
        if self._image_reward_model is None:
            try:
                # Patch: newer transformers moved these to pytorch_utils
                import transformers.modeling_utils as _mu
                try:
                    from transformers.pytorch_utils import (
                        apply_chunking_to_forward,
                        find_pruneable_heads_and_indices,
                        prune_linear_layer,
                    )
                    for _name, _fn in [
                        ("apply_chunking_to_forward", apply_chunking_to_forward),
                        ("find_pruneable_heads_and_indices", find_pruneable_heads_and_indices),
                        ("prune_linear_layer", prune_linear_layer),
                    ]:
                        if not hasattr(_mu, _name):
                            setattr(_mu, _name, _fn)
                except ImportError:
                    pass

                import ImageReward as RM

                self._image_reward_model = RM.load("ImageReward-v1.0", device=self.device)
                print("[Metrics] ImageReward loaded (ImageReward-v1.0)")
            except ImportError as e:
                print(f"[Warning] ImageReward not found, will be skipped: {e}")
                print("  Install: pip install image-reward")
                self._image_reward_model = False  # Mark as failed
            except Exception as e:
                print(f"[Warning] ImageReward load failed: {e}")
                self._image_reward_model = False  # Mark as failed
        return self._image_reward_model if self._image_reward_model is not False else None

    def compute_image_reward(self, image: Image.Image, text: str) -> Optional[float]:
        """Compute ImageReward score. Higher is better."""
        model = self._get_image_reward_model()
        if model is None:
            return None

        try:
            score = model.score(text, image)
            return float(score)
        except Exception as e:
            # Only print once per session
            if not hasattr(self, "_ir_error_printed"):
                print(f"[Warning] ImageReward compute failed: {e}")
                self._ir_error_printed = True
            return None

    # ==================== HPSv2 ====================

    def _get_hps_module(self):
        """
        Lazy load HPSv2 module.

        Reference: https://github.com/tgxs002/HPSv2
        Set HPS_ROOT environment variable for custom cache path.
        """
        if self._hps_module is None:
            try:
                import hpsv2

                self._hps_module = hpsv2
                print("[Metrics] HPSv2 loaded")
            except ImportError:
                print("[Warning] HPSv2 not found, will be skipped")
                print("  Install: pip install hpsv2")
                print("  Set cache: export HPS_ROOT=/your/cache/path")
                self._hps_module = False  # Mark as failed
        return self._hps_module if self._hps_module is not False else None

    def compute_hps(self, image: Image.Image, text: str) -> Optional[float]:
        """Compute HPS v2.1 score. Higher is better."""
        hps = self._get_hps_module()
        if hps is None:
            return None

        try:
            scores = hps.score(image, text, hps_version="v2.1")
            return float(scores[0]) if scores else None
        except Exception as e:
            # Only print once per session
            if not hasattr(self, "_hps_error_printed"):
                print(f"[Warning] HPS compute failed: {e}")
                self._hps_error_printed = True
            return None

    # ==================== Aesthetic Score (LAION) ====================

    def _get_aesthetic_scorer(self):
        """
        Lazy load LAION Aesthetic Predictor.

        Uses CLIP ViT-L/14 embeddings + MLP head.
        Weights: sac+logos+ava1-l14-linearMSE.pth
        Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor
        """
        if self._aesthetic_mlp is None:
            try:
                # Ensure CLIP is loaded (reuse existing)
                model, processor = self._get_clip_model("openai/clip-vit-large-patch14")
                if model is None:
                    self._aesthetic_mlp = False
                    return None

                # Download and load MLP weights
                state_dict = torch.hub.load_state_dict_from_url(
                    _AESTHETIC_WEIGHTS_URL, map_location=self.device
                )
                mlp = _AestheticMLP()
                mlp.load_state_dict(state_dict)
                mlp.eval().to(self.device)
                self._aesthetic_mlp = mlp
                print("[Metrics] Aesthetic Scorer loaded (LAION sac+logos+ava1-l14-linearMSE)")
            except Exception as e:
                print(f"[Warning] Aesthetic Scorer load failed: {e}")
                self._aesthetic_mlp = False
        return self._aesthetic_mlp if self._aesthetic_mlp is not False else None

    def compute_aesthetic_score(self, image: Image.Image) -> Optional[float]:
        """Compute LAION aesthetic score (1-10 scale). Higher is better."""
        mlp = self._get_aesthetic_scorer()
        if mlp is None:
            return None

        model, processor = self._get_clip_model("openai/clip-vit-large-patch14")
        if model is None:
            return None

        try:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embed = model.get_image_features(**inputs)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                score = mlp(embed).squeeze().item()

            return score
        except Exception as e:
            if not hasattr(self, "_aes_error_printed"):
                print(f"[Warning] Aesthetic score compute failed: {e}")
                self._aes_error_printed = True
            return None

    # ==================== DINOv2 (for Diversity) ====================

    def _get_dino_model(self):
        """
        Lazy load DINOv2-ViT-Large model.

        Reference: https://github.com/facebookresearch/dinov2
        Model: facebook/dinov2-large
        """
        if self._dino_model is None:
            try:
                from transformers import AutoImageProcessor, AutoModel

                self._dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", local_files_only=self.local_files_only)
                self._dino_model = AutoModel.from_pretrained("facebook/dinov2-large", local_files_only=self.local_files_only).eval().to(self.device)
                print("[Metrics] DINOv2 loaded (facebook/dinov2-large)")
            except ImportError:
                print("[Warning] transformers not found, DINOv2 diversity will be skipped")
                print("  Install: pip install transformers")
                self._dino_model = False
            except Exception as e:
                print(f"[Warning] DINOv2 load failed: {e}")
                self._dino_model = False
        return (
            (self._dino_model, self._dino_processor)
            if self._dino_model is not False
            else (None, None)
        )

    def _extract_features(
        self, images: list, model_type: str = "dino"
    ) -> Optional[torch.Tensor]:
        """
        Extract features from images using DINOv2 or CLIP.

        Args:
            images: List of PIL Images
            model_type: "dino" or "clip"

        Returns:
            Tensor of shape [N, D] with L2-normalized features
        """
        if not images:
            return None

        if model_type == "dino":
            model, processor = self._get_dino_model()
            if model is None:
                return None

            try:
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use CLS token embedding
                    features = outputs.last_hidden_state[:, 0, :]
                    features = features / features.norm(dim=-1, keepdim=True)
                return features
            except Exception as e:
                print(f"[Warning] DINOv2 feature extraction failed: {e}")
                return None

        elif model_type == "clip":
            model, processor = self._get_clip_model()
            if model is None:
                return None

            try:
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                return features
            except Exception as e:
                print(f"[Warning] CLIP feature extraction failed: {e}")
                return None

        return None

    def compute_diversity(
        self, images: list, model_type: str = "dino"
    ) -> Optional[float]:
        """
        Compute diversity score for a group of images.

        Formula: Diversity = 1 - (2 / L(L-1)) * sum(cos_sim(x_i, x_j))
        where L = number of images.

        Args:
            images: List of PIL Images (must have >= 2 images)
            model_type: "dino" or "clip"

        Returns:
            Diversity score in [0, 1], higher means more diverse
        """
        L = len(images)
        if L < 2:
            return None

        features = self._extract_features(images, model_type)
        if features is None:
            return None

        # Compute pairwise cosine similarity
        # features: [L, D], already L2-normalized
        sim_matrix = features @ features.T  # [L, L]

        # Sum upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(L, L, device=self.device), diagonal=1)
        total_sim = (sim_matrix * mask).sum().item()

        # Apply formula: 1 - (2 / L(L-1)) * sum(cos_sim)
        diversity = 1.0 - (2.0 / (L * (L - 1))) * total_sim

        return diversity


# ============================================================
# Main Function
# ============================================================


def compute_metrics(
    input: str,
    task_type: str = "text2svg",
    image_base_dir: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    output: Optional[str] = None,
    render_size: int = 512,
    device: str = "cuda",
    metrics: str = "basic,clip,preference",
    clip_model: str = "openai/clip-vit-large-patch14",
    lpips_network: str = "vgg",
    diversity_model: str = "dino",
    max_samples: Optional[int] = None,
    caption_dataset: Optional[str] = None,
    alpha_background: str = "white",
    local_files_only: bool = False,
):
    """
    Compute evaluation metrics from JSONL results.

    Args:
        input: Path to input JSONL file (output from hivg_metric.eval)
        task_type: "img2svg" or "text2svg"
        image_base_dir: Base directory for input images (required for img2svg visual)
        dataset: Dataset name in dataset_info.json (for img2svg visual metrics)
        dataset_dir: Directory containing dataset_info.json (for img2svg visual metrics)
        output: Path to output JSON file (optional)
        render_size: Size to render SVG for comparison
        device: Device for computation ("cuda" or "cpu")
        metrics: Comma-separated list of metrics to compute:
            - basic: Success rate, SVG length, path count
            - visual: SSIM, LPIPS, PSNR (img2svg only, requires dataset/dataset_dir)
            - clip: CLIP score (text2svg only)
            - pick: PickScore (both)
            - image_reward: ImageReward score (text+image)
            - hps: HPS v2.1 score (text+image)
            - aesthetic_score: LAION Aesthetic Score (image-only)
            - preference: Shorthand for pick + image_reward + hps + aesthetic_score
            - diversity: Sample diversity per prompt group (both)
            - all: All applicable metrics
        clip_model: CLIP model name (default: openai/clip-vit-large-patch14)
        lpips_network: LPIPS backbone network: "alex" (faster) or "vgg" (more accurate)
        diversity_model: Model for diversity computation: "dino" or "clip" (default: dino)
        max_samples: Maximum samples to process (for testing)
        caption_dataset: Path to HuggingFace Arrow dataset for img2svg caption lookup.
            Must contain 'md5' and 'long_caption' fields. Image filename stem is used
            as MD5 key. Enables CLIP/PickScore/ImageReward/HPS for img2svg.
        alpha_background: Background color used when compositing RGBA images for metric
            computation. Options: "white", "black", or hex like "#RRGGBB".
    """
    if task_type not in ["img2svg", "text2svg"]:
        raise ValueError(f"task_type must be 'img2svg' or 'text2svg', got '{task_type}'")

    background_rgb = _parse_background_color(alpha_background)

    # Parse metrics (handle both string and tuple from fire)
    if isinstance(metrics, (list, tuple)):
        metrics_list = [m.strip().lower() for m in metrics]
    else:
        metrics_list = [m.strip().lower() for m in metrics.split(",")]
    if "all" in metrics_list:
        if task_type == "img2svg":
            metrics_list = ["basic", "visual", "pick", "image_reward", "hps", "aesthetic_score"]
        else:
            metrics_list = ["basic", "clip", "pick", "image_reward", "hps", "aesthetic_score", "diversity"]

    # Expand "preference" alias into four independent sub-metrics
    if "preference" in metrics_list:
        metrics_list.remove("preference")
        for sub in ["pick", "image_reward", "hps", "aesthetic_score"]:
            if sub not in metrics_list:
                metrics_list.append(sub)

    compute_basic = "basic" in metrics_list
    compute_visual = "visual" in metrics_list and task_type == "img2svg"
    compute_clip = "clip" in metrics_list and task_type == "text2svg"
    compute_pick = "pick" in metrics_list
    compute_image_reward = "image_reward" in metrics_list
    compute_hps = "hps" in metrics_list
    compute_aesthetic_score = "aesthetic_score" in metrics_list
    compute_diversity = "diversity" in metrics_list and task_type == "text2svg"

    # Load caption lookup for img2svg text-dependent metrics
    caption_lookup = {}
    _needs_caption = task_type == "img2svg" and (compute_pick or compute_image_reward or compute_hps)
    if caption_dataset and task_type == "img2svg":
        if not image_base_dir:
            raise ValueError("--caption_dataset requires --image_base_dir to resolve image paths")
        caption_lookup = _load_caption_lookup(caption_dataset)
    elif _needs_caption:
        print("[Warning] img2svg with pick/image_reward/hps metrics requires --caption_dataset for text prompts; "
              "these text-dependent metrics will be skipped for samples without captions")

    # Validate diversity_model parameter
    if compute_diversity and diversity_model not in ["dino", "clip", "both"]:
        raise ValueError(f"diversity_model must be 'dino', 'clip', or 'both', got '{diversity_model}'")

    # Validate visual metrics requirements
    if compute_visual:
        if image_base_dir is None:
            raise ValueError("img2svg visual metrics require --image_base_dir")
        if dataset is None or dataset_dir is None:
            raise ValueError("img2svg visual metrics require --dataset and --dataset_dir")

    # Load image paths from original dataset (for img2svg visual metrics)
    dataset_image_paths = []
    if compute_visual and dataset and dataset_dir:
        dataset_image_paths = _load_image_paths_from_dataset(dataset, dataset_dir)

    # Load results
    results = []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if max_samples:
        results = results[:max_samples]

    # Validate prompt_index for diversity computation (fail-fast)
    if compute_diversity:
        has_prompt_index = any(r.get("prompt_index") is not None for r in results)
        if not has_prompt_index:
            raise ValueError(
                "Diversity computation requires 'prompt_index' field in input data, "
                "but no records contain this field. "
                "Ensure your inference output includes prompt_index for grouping samples."
            )

    print(f"[Metrics] Loaded {len(results)} samples from {input}")
    print(f"[Metrics] Task type: {task_type}")
    print(f"[Metrics] Device: {device}")
    print(f"[Metrics] Metrics: {', '.join(metrics_list)}")
    print(f"[Metrics] Alpha background: {alpha_background}")
    if compute_clip:
        print(f"[Metrics] CLIP model: {clip_model}")
    if compute_diversity:
        print(f"[Metrics] Diversity model: {diversity_model}")
    print()

    # Initialize computer
    computer = MetricsComputer(device=device, render_size=render_size, lpips_network=lpips_network, local_files_only=local_files_only)

    # Compute metrics
    metrics_result = MetricsResult()
    metrics_result.total_samples = len(results)

    # Collectors
    svg_char_lengths = []
    svg_token_counts = []
    path_counts = []
    path_commands_counts = []
    ssim_scores = []
    lpips_scores = []
    psnr_scores = []
    clip_visual_sim_scores = []
    clip_scores = []
    pick_scores = []
    image_reward_scores = []
    hps_scores = []
    aesthetic_scores = []

    # Cache rendered images for diversity computation (avoid re-rendering)
    rendered_images_cache = {}  # index -> PIL Image

    # Caption lookup miss tracking (img2svg only)
    _caption_miss_count = 0
    _caption_hit_count = 0
    _visual_resolved_count = 0
    _visual_missing_count = 0
    _visual_source_counter = defaultdict(int)

    for record in tqdm(results, desc="Computing metrics"):
        idx = record.get("index", 0)
        prompt = record.get("prompt", "")
        svg = record.get("svg", "")
        success = record.get("success", False)

        sample_metrics = {"index": idx, "success": success}

        if not success or not svg:
            metrics_result.per_sample.append(sample_metrics)
            continue

        metrics_result.success_count += 1

        # Basic stats
        if compute_basic:
            char_len = len(svg)
            token_count = _count_svg_tokens(svg)
            path_count = _count_svg_paths(svg)
            cmd_count = _count_path_commands(svg)

            svg_char_lengths.append(char_len)
            svg_token_counts.append(token_count)
            path_counts.append(path_count)
            path_commands_counts.append(cmd_count)

            sample_metrics["svg_char_length"] = char_len
            sample_metrics["svg_token_count"] = token_count
            sample_metrics["path_count"] = path_count
            sample_metrics["path_commands_count"] = cmd_count

        # Render predicted SVG
        pred_img = _render_svg_to_pil(svg, render_size, background_rgb=background_rgb)
        if pred_img is None:
            sample_metrics["render_success"] = False
            metrics_result.per_sample.append(sample_metrics)
            continue

        # Render success
        metrics_result.render_success_count += 1
        sample_metrics["render_success"] = True

        # Cache rendered image for diversity computation
        if compute_diversity:
            rendered_images_cache[idx] = pred_img

        # ===== Resolve input image path (for visual metrics and caption lookup) =====
        input_img_path = None
        input_img_source = None
        if task_type == "img2svg" and (compute_visual or caption_lookup):
            input_img_path = _resolve_existing_image_path(record.get("image_path"), image_base_dir=image_base_dir)
            if input_img_path:
                input_img_source = "record"
            if not input_img_path:
                input_img_path = _extract_image_path(prompt, image_base_dir or "")
                if input_img_path:
                    input_img_source = "prompt"
            if not input_img_path and dataset_image_paths:
                prompt_idx = record.get("prompt_index", idx)
                if isinstance(prompt_idx, int) and 0 <= prompt_idx < len(dataset_image_paths) and dataset_image_paths[prompt_idx]:
                    input_img_path = _resolve_existing_image_path(dataset_image_paths[prompt_idx], image_base_dir=image_base_dir)
                    if input_img_path:
                        input_img_source = "dataset"

            if input_img_path:
                _visual_resolved_count += 1
                _visual_source_counter[input_img_source] += 1
                sample_metrics["input_image_path"] = input_img_path
                sample_metrics["input_image_source"] = input_img_source
            else:
                _visual_missing_count += 1

        # ===== img2svg: Visual Similarity =====
        if compute_visual and image_base_dir:
            if input_img_path:
                gt_img = _to_rgb_with_background(Image.open(input_img_path), background_rgb=background_rgb)
                visual_metrics = computer.compute_image_metrics(pred_img, gt_img)

                if "ssim" in visual_metrics:
                    ssim_scores.append(visual_metrics["ssim"])
                    sample_metrics["ssim"] = visual_metrics["ssim"]
                if "lpips" in visual_metrics:
                    lpips_scores.append(visual_metrics["lpips"])
                    sample_metrics["lpips"] = visual_metrics["lpips"]
                if "psnr" in visual_metrics:
                    psnr_scores.append(visual_metrics["psnr"])
                    sample_metrics["psnr"] = visual_metrics["psnr"]

                # CLIP visual similarity (image-image cosine sim)
                clip_vis_sim = computer.compute_clip_visual_similarity(pred_img, gt_img, clip_model)
                if clip_vis_sim is not None:
                    clip_visual_sim_scores.append(clip_vis_sim)
                    sample_metrics["clip_visual_sim"] = clip_vis_sim

        # Extract text prompt for text-based metrics
        if task_type == "text2svg":
            text_prompt = _extract_prompt_text(prompt)
        elif caption_lookup and input_img_path:
            stem_key = _get_filename_stem(input_img_path)
            text_prompt = caption_lookup.get(stem_key, "")
            if text_prompt:
                _caption_hit_count += 1
            else:
                _caption_miss_count += 1
        else:
            text_prompt = ""

        # ===== CLIP Score =====
        if compute_clip and text_prompt:
            clip_score = computer.compute_clip_score(pred_img, text_prompt, clip_model)
            if clip_score is not None:
                clip_scores.append(clip_score)
                sample_metrics["clip_score"] = clip_score

        # ===== PickScore =====
        if compute_pick and text_prompt:
            pick_score = computer.compute_pick_score(pred_img, text_prompt)
            if pick_score is not None:
                pick_scores.append(pick_score)
                sample_metrics["pick_score"] = pick_score

        # ===== ImageReward =====
        if compute_image_reward and text_prompt:
            ir_score = computer.compute_image_reward(pred_img, text_prompt)
            if ir_score is not None:
                image_reward_scores.append(ir_score)
                sample_metrics["image_reward"] = ir_score

        # ===== HPS =====
        if compute_hps and text_prompt:
            hps_score = computer.compute_hps(pred_img, text_prompt)
            if hps_score is not None:
                hps_scores.append(hps_score)
                sample_metrics["hps"] = hps_score

        # ===== Aesthetic Score =====
        if compute_aesthetic_score:
            aes_score = computer.compute_aesthetic_score(pred_img)
            if aes_score is not None:
                aesthetic_scores.append(aes_score)
                sample_metrics["aesthetic_score"] = aes_score

        metrics_result.per_sample.append(sample_metrics)

    # Caption lookup summary (img2svg only)
    if caption_lookup and (_caption_hit_count + _caption_miss_count) > 0:
        total_looked = _caption_hit_count + _caption_miss_count
        print(f"[Metrics] Caption lookup: {_caption_hit_count}/{total_looked} hit, "
              f"{_caption_miss_count} missed (IR/HPS/PickScore skipped for misses)")

    # Input image path resolution summary (img2svg only)
    if task_type == "img2svg" and (compute_visual or caption_lookup):
        looked_up = _visual_resolved_count + _visual_missing_count
        if looked_up > 0:
            source_parts = []
            for source_name in ["record", "prompt", "dataset"]:
                source_parts.append(f"{source_name}={_visual_source_counter.get(source_name, 0)}")
            print(
                f"[Metrics] Input image path resolved: {_visual_resolved_count}/{looked_up} "
                f"(missing={_visual_missing_count}, {', '.join(source_parts)})"
            )

    # ===== Diversity Computation (per-prompt group) =====
    diversity_dino_scores = []
    diversity_clip_scores = []

    if compute_diversity:
        prompt_groups = defaultdict(list)  # group_key -> list of (index, record)

        print("[Metrics] Using prompt_index for diversity grouping")

        for record in results:
            if not record.get("success", False):
                continue

            record_index = record.get("index")
            if record_index is None or record_index not in rendered_images_cache:
                continue

            group_key = record.get("prompt_index")
            if group_key is None:
                continue

            prompt_groups[group_key].append(record_index)

        # Compute diversity for each group with >= 2 samples
        valid_groups = [(k, indices) for k, indices in prompt_groups.items() if len(indices) >= 2]

        if valid_groups:
            print(f"\n[Metrics] Computing diversity for {len(valid_groups)} prompt groups...")

            for group_key, indices in tqdm(valid_groups, desc="Computing diversity"):
                # Use cached images instead of re-rendering
                images = [rendered_images_cache[idx] for idx in indices]

                if diversity_model == "dino":
                    score = computer.compute_diversity(images, model_type="dino")
                    if score is not None:
                        diversity_dino_scores.append(score)
                elif diversity_model == "clip":
                    score = computer.compute_diversity(images, model_type="clip")
                    if score is not None:
                        diversity_clip_scores.append(score)
                else:  # "both"
                    dino_score = computer.compute_diversity(images, model_type="dino")
                    if dino_score is not None:
                        diversity_dino_scores.append(dino_score)
                    clip_score = computer.compute_diversity(images, model_type="clip")
                    if clip_score is not None:
                        diversity_clip_scores.append(clip_score)
        else:
            print("[Warning] No prompt groups with >= 2 samples for diversity computation")

        # Clear cache after diversity computation
        rendered_images_cache.clear()

    # Aggregate metrics
    metrics_result.success_rate = (
        metrics_result.success_count / metrics_result.total_samples
        if metrics_result.total_samples > 0
        else 0
    )
    metrics_result.render_success_rate = (
        metrics_result.render_success_count / metrics_result.total_samples
        if metrics_result.total_samples > 0
        else 0
    )
    metrics_result.avg_svg_char_length = float(np.mean(svg_char_lengths)) if svg_char_lengths else 0
    metrics_result.avg_svg_token_count = float(np.mean(svg_token_counts)) if svg_token_counts else 0
    metrics_result.avg_path_count = float(np.mean(path_counts)) if path_counts else 0
    metrics_result.avg_path_commands_count = float(np.mean(path_commands_counts)) if path_commands_counts else 0

    if ssim_scores:
        metrics_result.ssim_mean = float(np.mean(ssim_scores))
        metrics_result.ssim_std = float(np.std(ssim_scores))
    if lpips_scores:
        metrics_result.lpips_mean = float(np.mean(lpips_scores))
        metrics_result.lpips_std = float(np.std(lpips_scores))
    if psnr_scores:
        metrics_result.psnr_mean = float(np.mean(psnr_scores))
        metrics_result.psnr_std = float(np.std(psnr_scores))
    if clip_visual_sim_scores:
        metrics_result.clip_visual_sim_mean = float(np.mean(clip_visual_sim_scores))
        metrics_result.clip_visual_sim_std = float(np.std(clip_visual_sim_scores))
    if clip_scores:
        metrics_result.clip_score_mean = float(np.mean(clip_scores))
        metrics_result.clip_score_std = float(np.std(clip_scores))
    if pick_scores:
        metrics_result.pick_score_mean = float(np.mean(pick_scores))
        metrics_result.pick_score_std = float(np.std(pick_scores))
    if image_reward_scores:
        metrics_result.image_reward_mean = float(np.mean(image_reward_scores))
        metrics_result.image_reward_std = float(np.std(image_reward_scores))
    if hps_scores:
        metrics_result.hps_mean = float(np.mean(hps_scores))
        metrics_result.hps_std = float(np.std(hps_scores))
    if aesthetic_scores:
        metrics_result.aesthetic_score_mean = float(np.mean(aesthetic_scores))
        metrics_result.aesthetic_score_std = float(np.std(aesthetic_scores))
    if diversity_dino_scores:
        metrics_result.diversity_dino_mean = float(np.mean(diversity_dino_scores))
        metrics_result.diversity_dino_std = float(np.std(diversity_dino_scores))
    if diversity_clip_scores:
        metrics_result.diversity_clip_mean = float(np.mean(diversity_clip_scores))
        metrics_result.diversity_clip_std = float(np.std(diversity_clip_scores))

    # Print results
    print("\n" + "=" * 60)
    print("HiVG Evaluation Metrics")
    print("=" * 60)

    print("\n[Basic Statistics]")
    print(f"  Total Samples:        {metrics_result.total_samples}")
    print(f"  Token Parse Success:  {metrics_result.success_count} ({metrics_result.success_rate:.2%})")
    print(f"  Render Success:       {metrics_result.render_success_count} ({metrics_result.render_success_rate:.2%})")
    if compute_basic:
        print(f"  Avg SVG Char Length:     {metrics_result.avg_svg_char_length:.1f}")
        print(f"  Avg SVG Token Count:     {metrics_result.avg_svg_token_count:.1f}")
        print(f"  Avg Path Count:          {metrics_result.avg_path_count:.1f}")
        print(f"  Avg Path Commands Count: {metrics_result.avg_path_commands_count:.1f}")

    if metrics_result.ssim_mean is not None or metrics_result.clip_visual_sim_mean is not None:
        print("\n[Visual Similarity] (img2svg)")
        print(f"  SSIM:   {metrics_result.ssim_mean:.4f} +/- {metrics_result.ssim_std:.4f} (higher is better)")
        if metrics_result.lpips_mean is not None:
            print(f"  LPIPS:  {metrics_result.lpips_mean:.4f} +/- {metrics_result.lpips_std:.4f} (lower is better)")
        if metrics_result.psnr_mean is not None:
            print(f"  PSNR:   {metrics_result.psnr_mean:.2f} +/- {metrics_result.psnr_std:.2f} dB (higher is better)")
        if metrics_result.clip_visual_sim_mean is not None:
            print(f"  CLIP Visual Sim: {metrics_result.clip_visual_sim_mean:.4f} +/- {metrics_result.clip_visual_sim_std:.4f} (higher is better)")

    if metrics_result.clip_score_mean is not None:
        print("\n[Semantic Alignment] (text2svg)")
        print(f"  CLIP Score: {metrics_result.clip_score_mean:.4f} +/- {metrics_result.clip_score_std:.4f}")

    if metrics_result.pick_score_mean is not None or metrics_result.image_reward_mean is not None or metrics_result.hps_mean is not None or metrics_result.aesthetic_score_mean is not None:
        print("\n[Aesthetic Quality / Human Preference]")
        if metrics_result.pick_score_mean is not None:
            print(f"  PickScore:        {metrics_result.pick_score_mean:.4f} +/- {metrics_result.pick_score_std:.4f}")
        if metrics_result.image_reward_mean is not None:
            print(f"  ImageReward:      {metrics_result.image_reward_mean:.4f} +/- {metrics_result.image_reward_std:.4f}")
        if metrics_result.hps_mean is not None:
            print(f"  HPS v2.1:         {metrics_result.hps_mean:.4f} +/- {metrics_result.hps_std:.4f}")
        if metrics_result.aesthetic_score_mean is not None:
            print(f"  Aesthetic Score:   {metrics_result.aesthetic_score_mean:.4f} +/- {metrics_result.aesthetic_score_std:.4f} (1-10 scale)")

    if metrics_result.diversity_dino_mean is not None or metrics_result.diversity_clip_mean is not None:
        print("\n[Diversity] (per-prompt group, higher is better)")
        if metrics_result.diversity_dino_mean is not None:
            print(f"  DINOv2: {metrics_result.diversity_dino_mean:.4f} +/- {metrics_result.diversity_dino_std:.4f}")
        if metrics_result.diversity_clip_mean is not None:
            print(f"  CLIP:   {metrics_result.diversity_clip_mean:.4f} +/- {metrics_result.diversity_clip_std:.4f}")

    print("=" * 60)

    # Save results
    if output:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(metrics_result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n[Metrics] Results saved to {output}")

    return metrics_result


if __name__ == "__main__":
    fire.Fire(compute_metrics)
