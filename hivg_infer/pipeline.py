"""
HiSVG Inference Pipeline.

Supports Qwen2-VL based models for both text2svg and img2svg tasks.
Optionally loads LoRA adapters via peft.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from .postprocess import postprocess_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _resolve_model_class(model_path: str, trust_remote_code: bool):
    """
    Select the correct HF model class based on the checkpoint's architecture.

    Vision-language checkpoints like Qwen2.5-VL cannot be loaded through
    ``AutoModelForCausalLM`` and require their task-specific class instead.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "") or ""

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration

    architectures = getattr(config, "architectures", None) or []
    if architectures:
        arch = architectures[0]
        try:
            import transformers as _tf
            return getattr(_tf, arch)
        except AttributeError:
            pass

    return AutoModelForCausalLM


def _build_vl_messages(
    messages: List[Dict],
    images: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Convert plain text messages + image path list to Qwen2-VL multimodal format.

    Qwen2-VL expects the user content to be a list of typed items:
        [{"type": "image", "image": <PIL.Image>}, {"type": "text", "text": "..."}]

    If no images are provided, messages are returned unchanged (text-only path).
    """
    if not images:
        return messages

    pil_images = [_load_pil_image(p) if isinstance(p, str) else p for p in images]

    result = []
    img_iter = iter(pil_images)

    for msg in messages:
        if msg["role"] == "user":
            content_str = msg["content"] if isinstance(msg["content"], str) else ""
            # Strip explicit <image> placeholder tokens from the text body
            clean_text = re.sub(r"<image>", "", content_str).strip()

            new_content: List[Dict] = []
            # Attach next available image (one image per user turn)
            try:
                new_content.append({"type": "image", "image": next(img_iter)})
            except StopIteration:
                pass

            if clean_text:
                new_content.append({"type": "text", "text": clean_text})

            result.append({"role": msg["role"], "content": new_content})
        else:
            result.append(msg)

    return result


def _extract_pil_images_from_messages(messages: List[Dict]) -> List[Image.Image]:
    """Walk multimodal messages and collect PIL images in order."""
    imgs = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image":
                    imgs.append(item["image"])
    return imgs


def _load_and_normalize_dataset(
    dataset: Union[str, List[dict]],
    fmt: str = "sharegpt",
) -> List[dict]:
    """Load a JSON dataset and normalise it to the internal sharegpt-like format."""
    import json

    if isinstance(dataset, str):
        with open(dataset, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = dataset

    if fmt == "sharegpt":
        return samples
    elif fmt == "alpaca":
        normalised = []
        for item in samples:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            user_content = f"{instruction}\n{inp}".strip() if inp else instruction
            msgs = [{"role": "user", "content": user_content}]
            if output:
                msgs.append({"role": "assistant", "content": output})
            normalised.append({
                "messages": msgs,
                "images": item.get("images", None),
                "_gt": output,
            })
        return normalised
    else:
        raise ValueError(f"Unknown format '{fmt}'. Use 'sharegpt' or 'alpaca'.")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Generation sampling parameters."""

    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 4096
    repetition_penalty: float = 1.0
    num_beams: int = 1

    def to_dict(self) -> dict:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_GENERATION_FIELDS = set(GenerationConfig.__dataclass_fields__.keys())

# Tokens added by chat templates that must be stripped from the decoded output
# while preserving HiVG special tokens (e.g. <svg>, <path>, <cmd_M>, ...).
_CHAT_END_TOKENS = ["<|im_end|>", "<|endoftext|>"]


class HiSVGInferencePipeline:
    """
    SVG generation inference pipeline built on HuggingFace transformers.

    Wraps a Qwen2-VL (or compatible) model, applies the chat template,
    decodes with skip_special_tokens=False (required to preserve HiVG tokens),
    then converts the token sequence to a valid SVG using hivg_tokenizer.

    Args:
        model_path:        Path to the model (checkpoint or HuggingFace hub ID).
        adapter_path:      Optional LoRA adapter path (requires ``peft``).
        coord_range:       SVG canvas coordinate range (default 234).
        relative_range:    Relative coordinate range passed to the detokenizer.
        device_map:        HuggingFace device-map strategy (default "auto").
        torch_dtype:       Model weight dtype — "bfloat16", "float16", or "float32".
        trust_remote_code: Allow custom model code from the checkpoint.
        **generation_kwargs: Any ``GenerationConfig`` field (temperature, top_p, …).
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        coord_range: int = 234,
        relative_range: Optional[int] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        **generation_kwargs,
    ):
        self.coord_range = coord_range
        self.relative_range = relative_range

        gen_kwargs = {k: v for k, v in generation_kwargs.items() if k in _GENERATION_FIELDS}
        self.generation_config = GenerationConfig(**gen_kwargs)

        self._load_model(model_path, adapter_path, device_map, torch_dtype, trust_remote_code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(
        self,
        model_path: str,
        adapter_path: Optional[str],
        device_map: str,
        torch_dtype: str,
        trust_remote_code: bool,
    ) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        model_cls = _resolve_model_class(model_path, trust_remote_code)
        self.model = model_cls.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "peft is required to load LoRA adapters. "
                    "Install with: pip install peft"
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            print(f"[Pipeline] LoRA adapter loaded and merged: {adapter_path}")

        self.model.eval()

    def _prepare_inputs(
        self,
        messages: List[Dict],
        images: Optional[List[str]] = None,
    ) -> dict:
        """Tokenise messages (+ images) and return model input tensors."""
        vl_messages = _build_vl_messages(messages, images)

        # Apply the model's native chat template
        text = self.processor.apply_chat_template(
            vl_messages, tokenize=False, add_generation_prompt=True
        )

        pil_images = _extract_pil_images_from_messages(vl_messages)

        if pil_images:
            inputs = self.processor(
                text=[text], images=pil_images, return_tensors="pt"
            )
        else:
            inputs = self.processor(text=[text], return_tensors="pt")

        # Move all tensor values to model device
        device = next(self.model.parameters()).device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _decode_output(self, output_ids: torch.Tensor, input_len: int) -> str:
        """Decode new tokens with special tokens preserved, then strip chat end tokens."""
        new_ids = output_ids[input_len:]
        raw = self.processor.decode(new_ids, skip_special_tokens=False)
        for tok in _CHAT_END_TOKENS:
            raw = raw.replace(tok, "")
        return raw.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_single(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> dict:
        """
        Generate SVG from a single conversation turn.

        Args:
            messages: List of message dicts, e.g.
                      ``[{"role": "user", "content": "Draw a red circle"}]``
            images:   Optional list of image file paths (for img2svg tasks).
            **kwargs: Override any ``GenerationConfig`` parameter for this call.

        Returns:
            dict with keys: ``svg``, ``tokens``, ``raw_output``, ``success``, ``error``
        """
        inputs = self._prepare_inputs(messages, images)
        input_len = inputs["input_ids"].shape[1]

        gen_params = {**self.generation_config.to_dict(), **kwargs}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_params)

        raw_output = self._decode_output(output_ids[0], input_len)
        return postprocess_output(raw_output, self.coord_range, self.relative_range)

    def text2svg(self, description: str, **kwargs) -> dict:
        """
        Generate SVG from a text description.

        The method automatically prepends the training instruction prefix
        ``[T2ST] Generate SVG tokens from text.`` so that ``description``
        only needs to be the plain natural-language text.

        Args:
            description: Natural-language description of the desired SVG.
            **kwargs:     Generation parameter overrides.

        Returns:
            Result dict with ``svg``, ``tokens``, ``success``, ``error``.
        """
        prompt = f"[T2ST] Generate SVG tokens from text.\n{description}"
        messages = [{"role": "user", "content": prompt}]
        return self.generate_single(messages, **kwargs)

    def img2svg(
        self,
        image_path: str,
        **kwargs,
    ) -> dict:
        """
        Generate SVG from an input image.

        Uses the standard training instruction
        ``[I2ST] Generate SVG tokens from image.`` followed by the image token,
        matching the format the model was trained on.

        Args:
            image_path: Path to the source image.
            **kwargs:   Generation parameter overrides.

        Returns:
            Result dict with ``svg``, ``tokens``, ``success``, ``error``.
        """
        prompt = "[I2ST] Generate SVG tokens from image.\n\n<image>"
        messages = [{"role": "user", "content": prompt}]
        return self.generate_single(messages, images=[image_path], **kwargs)

    def generate_batch(
        self,
        dataset: Union[str, List[dict]],
        output_dir: Optional[str] = None,
        save_svg: bool = True,
        save_tokens: bool = False,
        max_samples: Optional[int] = None,
        fmt: str = "sharegpt",
    ) -> List[dict]:
        """
        Run batch inference on a dataset file or list of samples.

        Args:
            dataset:     Path to a JSON file or list of sample dicts.
                         Supports ShareGPT format (default) and Alpaca format.
            output_dir:  Directory to write individual ``.svg`` files.
            save_svg:    Whether to save SVG files to ``output_dir``.
            save_tokens: Whether to save raw token files alongside SVGs.
            max_samples: Cap the number of samples processed.
            fmt:         ``"sharegpt"`` (default) or ``"alpaca"``.

        Returns:
            List of result dicts (one per sample).
        """
        samples = _load_and_normalize_dataset(dataset, fmt=fmt)

        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]

        if output_dir and save_svg:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for idx, sample in enumerate(samples):
            messages = sample.get("messages", [])
            images = sample.get("images") or None

            result = self.generate_single(messages, images)
            result["index"] = idx

            if output_dir and result["success"]:
                if save_svg and result["svg"]:
                    svg_path = os.path.join(output_dir, f"{idx:06d}.svg")
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(result["svg"])
                    result["svg_path"] = svg_path

                if save_tokens and result["tokens"]:
                    tok_path = os.path.join(output_dir, f"{idx:06d}.tokens")
                    with open(tok_path, "w", encoding="utf-8") as f:
                        f.write(result["tokens"])
                    result["tokens_path"] = tok_path

            results.append(result)

        return results
