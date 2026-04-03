"""
HiVG Evaluation Script

Batch inference for SVG generation evaluation.
Supports both HuggingFace and vLLM backends.

Dataset format
--------------
Alpaca (text2svg):
    [{"instruction": "...", "input": "", "output": "<svg tokens>", "images": []}, ...]

ShareGPT (img2svg):
    [{"messages": [{"role": "user", "content": "<image>..."}],
      "images": ["path/to/image.png"],
      "label": "<svg tokens>"}, ...]

Usage
-----
# HuggingFace backend
python -m hivg_metric.eval \\
    --model_path /path/to/model \\
    --dataset test.json \\
    --format alpaca \\
    --coord_range 234 \\
    --save_name results.jsonl

# vLLM backend (faster batch inference)
python -m hivg_metric.eval \\
    --model_path /path/to/model \\
    --dataset test.json \\
    --infer_backend vllm \\
    --batch_size 32 \\
    --save_name results.jsonl \\
    --save_svg_dir ./svgs/
"""

import gc
import json
import os
import re
import sys
import time
from typing import Optional

import fire
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hivg_infer.postprocess import postprocess_output


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: str, fmt: str, max_samples: Optional[int] = None) -> list:
    """
    Load and normalise a dataset JSON file.

    Returns a list of dicts with keys:
        messages  – list of {"role", "content"} dicts (user turn only)
        images    – list of image paths (may be empty / None)
        label     – ground-truth SVG token string (may be empty)
        image_path – first image path string for record keeping
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if max_samples:
        raw = raw[:max_samples]

    samples = []
    for item in raw:
        if fmt == "alpaca":
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            label = item.get("output", "")
            user_content = f"{instruction}\n{inp}".strip() if inp else instruction
            messages = [{"role": "user", "content": user_content}]
            images = item.get("images") or []
            image_path = images[0] if images else None

        elif fmt == "sharegpt":
            messages_raw = item.get("messages", [])
            messages = [m for m in messages_raw if m["role"] != "assistant"]
            # GT is the last assistant turn (if present)
            label = ""
            for m in messages_raw:
                if m["role"] == "assistant":
                    label = m.get("content", "")
            # Also accept top-level "label" field
            label = item.get("label", label)
            images = item.get("images") or []
            image_path = images[0] if images else None

        else:
            raise ValueError(f"Unknown format '{fmt}'. Use 'alpaca' or 'sharegpt'.")

        samples.append({
            "messages": messages,
            "images": images if images else None,
            "label": label,
            "image_path": image_path,
        })

    return samples


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

def _run_hf_inference(
    model_path: str,
    samples: list,
    coord_range: int,
    relative_range: Optional[int],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
    adapter_path: Optional[str],
    torch_dtype: str,
    num_samples_per_prompt: int,
) -> tuple:
    from hivg_infer.pipeline import HiSVGInferencePipeline

    pipeline = HiSVGInferencePipeline(
        model_path=model_path,
        adapter_path=adapter_path,
        coord_range=coord_range,
        relative_range=relative_range,
        torch_dtype=torch_dtype,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )

    all_prompts, all_raw_outputs, all_tokens, all_svgs = [], [], [], []
    all_labels, all_label_svgs, all_success = [], [], []
    all_sample_ids, all_prompt_indices, all_image_paths = [], [], []

    for prompt_idx, sample in enumerate(tqdm(samples, desc="HuggingFace inference")):
        messages = sample["messages"]
        images = sample["images"]
        label = sample["label"]
        image_path = sample["image_path"]

        # Decode GT label to SVG
        label_result = postprocess_output(label, coord_range=coord_range, relative_range=relative_range)
        label_svg = label_result.get("svg", "")

        # Encode prompt text for record keeping
        prompt_text = " | ".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        for sample_id in range(num_samples_per_prompt):
            result = pipeline.generate_single(messages, images)

            all_prompts.append(prompt_text)
            all_raw_outputs.append(result.get("raw_output", ""))
            all_tokens.append(result.get("tokens", ""))
            all_svgs.append(result.get("svg", ""))
            all_labels.append(label)
            all_label_svgs.append(label_svg)
            all_success.append(result.get("success", False))
            all_sample_ids.append(sample_id)
            all_prompt_indices.append(prompt_idx)
            all_image_paths.append(image_path)

        gc.collect()

    return (
        all_prompts, all_raw_outputs, all_tokens, all_svgs,
        all_labels, all_label_svgs, all_success,
        all_sample_ids, all_prompt_indices, all_image_paths,
    )


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def _run_vllm_inference(
    model_path: str,
    samples: list,
    coord_range: int,
    relative_range: Optional[int],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
    batch_size: int,
    seed: Optional[int],
    num_samples_per_prompt: int,
    image_max_pixels: int,
    image_min_pixels: int,
) -> tuple:
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Determine tensor parallel size
    try:
        import torch
        tp = max(torch.cuda.device_count(), 1)
    except Exception:
        tp = 1

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 4},
    )

    sampling_params = SamplingParams(
        n=num_samples_per_prompt,
        temperature=temperature,
        top_p=top_p if top_p < 1.0 else 1.0,
        top_k=top_k if top_k > 0 else -1,
        repetition_penalty=repetition_penalty or 1.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
        seed=seed,
    )

    all_prompts, all_raw_outputs, all_tokens, all_svgs = [], [], [], []
    all_labels, all_label_svgs, all_success = [], [], []
    all_sample_ids, all_prompt_indices, all_image_paths = [], [], []

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="vLLM inference"):
        batch = samples[batch_start: batch_start + batch_size]
        vllm_inputs = []
        prompts_txt = []
        labels_batch = []
        img_paths_batch = []

        for sample in batch:
            messages = sample["messages"]
            images = sample["images"]
            label = sample["label"]
            image_path = sample["image_path"]

            # Build multimodal messages for chat template
            if images:
                from PIL import Image as PILImage
                pil_imgs = [PILImage.open(p).convert("RGB") for p in images]
                vl_messages = []
                img_iter = iter(pil_imgs)
                for msg in messages:
                    if msg["role"] == "user":
                        clean_text = re.sub(r"<image>", "", msg["content"]).strip()
                        content = []
                        try:
                            content.append({"type": "image", "image": next(img_iter)})
                        except StopIteration:
                            pass
                        if clean_text:
                            content.append({"type": "text", "text": clean_text})
                        vl_messages.append({"role": msg["role"], "content": content})
                    else:
                        vl_messages.append(msg)

                text = processor.apply_chat_template(
                    vl_messages, tokenize=False, add_generation_prompt=True
                )
                processed = processor(text=[text], images=pil_imgs, return_tensors="pt")
                mm_data = {"image": pil_imgs}
            else:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processed = processor(text=[text], return_tensors="pt")
                mm_data = None

            vllm_inputs.append({
                "prompt_token_ids": processed["input_ids"][0].tolist(),
                "multi_modal_data": mm_data,
            })
            prompts_txt.append(text)
            labels_batch.append(label)
            img_paths_batch.append(image_path)

        results = llm.generate(vllm_inputs, sampling_params)

        for i, result in enumerate(results):
            label = labels_batch[i]
            image_path = img_paths_batch[i]
            prompt_text = prompts_txt[i]
            prompt_idx = batch_start + i

            label_result = postprocess_output(label, coord_range=coord_range, relative_range=relative_range)
            label_svg = label_result.get("svg", "")

            for sample_id, output in enumerate(result.outputs):
                raw_output = output.text.replace("<|im_end|>", "").strip()
                pp = postprocess_output(raw_output, coord_range=coord_range, relative_range=relative_range)

                all_prompts.append(prompt_text)
                all_raw_outputs.append(raw_output)
                all_tokens.append(pp.get("tokens", ""))
                all_svgs.append(pp.get("svg", ""))
                all_labels.append(label)
                all_label_svgs.append(label_svg)
                all_success.append(pp.get("success", False))
                all_sample_ids.append(sample_id)
                all_prompt_indices.append(prompt_idx)
                all_image_paths.append(image_path)

        gc.collect()

    return (
        all_prompts, all_raw_outputs, all_tokens, all_svgs,
        all_labels, all_label_svgs, all_success,
        all_sample_ids, all_prompt_indices, all_image_paths,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def hivg_eval(
    model_path: str,
    dataset: str,
    format: str = "alpaca",
    infer_backend: str = "huggingface",
    save_name: str = "hivg_eval_results.jsonl",
    save_svg_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    coord_range: int = 234,
    relative_range: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 4096,
    repetition_penalty: float = 1.0,
    batch_size: int = 32,
    seed: Optional[int] = None,
    adapter_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    num_samples_per_prompt: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
):
    """
    Batch inference evaluation for HiVG.

    Args:
        model_path:             Path to the model checkpoint.
        dataset:                Path to the dataset JSON file.
        format:                 Dataset format: "alpaca" or "sharegpt".
        infer_backend:          "huggingface" or "vllm".
        save_name:              Output JSONL file path.
        save_svg_dir:           Directory to save individual SVG files (pred/ and gt/).
        max_samples:            Cap the number of samples to evaluate.
        coord_range:            HiVG coordinate range (default: 234).
        relative_range:         HiVG relative coordinate range.
        temperature:            Sampling temperature.
        top_p:                  Top-p (nucleus) sampling.
        top_k:                  Top-k sampling.
        max_new_tokens:         Maximum new tokens to generate.
        repetition_penalty:     Repetition penalty factor.
        batch_size:             Batch size (used by vLLM backend).
        seed:                   Random seed (vLLM only).
        adapter_path:           LoRA adapter path (HF backend only, requires peft).
        torch_dtype:            Model weight dtype (HF backend).
        num_samples_per_prompt: Number of outputs per prompt (>1 for diversity eval).
        image_max_pixels:       vLLM image pixel cap (max side²).
        image_min_pixels:       vLLM image minimum pixel count.
    """
    print(f"[Eval] Backend : {infer_backend}")
    print(f"[Eval] Dataset : {dataset} (format={format})")
    print(f"[Eval] Model   : {model_path}")
    print(f"[Eval] coord_range: {coord_range}")
    if num_samples_per_prompt > 1:
        print(f"[Eval] Samples per prompt: {num_samples_per_prompt} (diversity mode)")

    samples = _load_dataset(dataset, fmt=format, max_samples=max_samples)
    print(f"[Eval] Loaded {len(samples)} samples")

    start_time = time.time()

    if infer_backend == "vllm":
        results_tuple = _run_vllm_inference(
            model_path=model_path,
            samples=samples,
            coord_range=coord_range,
            relative_range=relative_range,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            seed=seed,
            num_samples_per_prompt=num_samples_per_prompt,
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
        )
    elif infer_backend == "huggingface":
        results_tuple = _run_hf_inference(
            model_path=model_path,
            samples=samples,
            coord_range=coord_range,
            relative_range=relative_range,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            adapter_path=adapter_path,
            torch_dtype=torch_dtype,
            num_samples_per_prompt=num_samples_per_prompt,
        )
    else:
        raise ValueError(f"Unknown infer_backend '{infer_backend}'. Use 'huggingface' or 'vllm'.")

    (
        all_prompts, all_raw_outputs, all_tokens, all_svgs,
        all_labels, all_label_svgs, all_success,
        all_sample_ids, all_prompt_indices, all_image_paths,
    ) = results_tuple

    elapsed = time.time() - start_time
    success_count = sum(all_success)
    total = len(all_success)
    print(f"[Eval] Done in {int(elapsed // 60)}m {elapsed % 60:.1f}s")
    print(f"[Eval] Success: {success_count}/{total} ({100 * success_count / max(total, 1):.2f}%)")

    # Save JSONL
    save_dir = os.path.dirname(save_name)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_name, "w", encoding="utf-8") as f:
        for idx, fields in enumerate(zip(
            all_prompts, all_raw_outputs, all_tokens, all_svgs,
            all_labels, all_label_svgs, all_success,
            all_sample_ids, all_prompt_indices, all_image_paths,
        )):
            (prompt, raw_output, tokens, svg, label, label_svg,
             success, sample_id, prompt_index, image_path) = fields
            record = {
                "index": idx,
                "prompt_index": prompt_index,
                "sample_id": sample_id,
                "image_path": image_path,
                "prompt": prompt,
                "raw_output": raw_output,
                "tokens": tokens,
                "svg": svg,
                "label": label,
                "label_svg": label_svg,
                "success": success,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Eval] Results saved to {save_name}")

    # Optionally save individual SVG files
    if save_svg_dir:
        pred_dir = os.path.join(save_svg_dir, "pred")
        gt_dir = os.path.join(save_svg_dir, "gt")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        for idx, (svg, label_svg, success) in enumerate(zip(all_svgs, all_label_svgs, all_success)):
            if success and svg:
                with open(os.path.join(pred_dir, f"{idx:06d}.svg"), "w", encoding="utf-8") as f:
                    f.write(svg)
            if label_svg:
                with open(os.path.join(gt_dir, f"{idx:06d}.svg"), "w", encoding="utf-8") as f:
                    f.write(label_svg)

        print(f"[Eval] SVG files saved to {save_svg_dir} (pred/ and gt/)")

    print("*" * 60)
    print(f"Total  : {total}")
    print(f"Success: {success_count} ({100 * success_count / max(total, 1):.2f}%)")
    print("*" * 60)


if __name__ == "__main__":
    fire.Fire(hivg_eval)
