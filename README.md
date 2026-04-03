<h1 align="center">HiVG: Hierarchical SVG Tokenization</h1>
<h3 align="center">Learning Compact Visual Programs for Scalable Vector Graphics Modeling</h3>

<p align="center">
    <a href="https://hy-hivg.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2506.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
    <a href="https://github.com/ximinng/HiVG"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GitHub"></a>
    <a href="https://github.com/ximinng/HiVG/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

<p align="center">
  <img src="assets/HiVG_tokenizer.gif" width="720" alt="HiVG Tokenizer">
</p>

## Highlights

- **State-of-the-Art SVG Generation** — 3B parameters, beats 7/7 proprietary models with +17% usability over GPT-5.
- **Hierarchical Tokenization** — Three-level tokenization (Raw SVG → Atomic tokens → BPE-merged segments) achieving 2.76x token compression.
- **Text-to-SVG & Image-to-SVG** — Unified pipeline supporting both `text2svg` and `img2svg` tasks with sub-second generation.

---

## Installation

```bash
git clone https://github.com/ximinng/HiVG.git
cd HiVG
pip install -e .
```

---

## Quick Start

### Python API

```python
from hivg_infer import HiSVGInferencePipeline

pipeline = HiSVGInferencePipeline(
    model_path="/path/to/model",
    coord_range=234,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=4096,
)

# Image-to-SVG
result = pipeline.img2svg("/path/to/photo.png")
if result["success"]:
    print(result["svg"])

# Text-to-SVG
result = pipeline.text2svg("A minimalist black phone icon with an outline style")
if result["success"]:
    with open("output.svg", "w") as f:
        f.write(result["svg"])
```

### Command Line

```bash
# Text-to-SVG
python -m hivg_infer.cli \
    --model_path /path/to/model \
    --prompt "A minimalist black phone icon with an outline style"

# Image-to-SVG
python -m hivg_infer.cli \
    --model_path /path/to/model \
    --image photo.png

# Batch inference (ShareGPT / Alpaca format)
python -m hivg_infer.cli \
    --model_path /path/to/model \
    --dataset test.json \
    --output_dir ./outputs

# Interactive mode
python -m hivg_infer.cli \
    --model_path /path/to/model \
    --interactive
```

---

## Evaluation

### Batch Inference

```bash
# HuggingFace backend
python -m hivg_metric.eval \
    --model_path /path/to/model \
    --dataset test.json \
    --format alpaca \
    --coord_range 234 \
    --save_name results.jsonl \
    --save_svg_dir ./svgs/

# vLLM backend (faster)
python -m hivg_metric.eval \
    --model_path /path/to/model \
    --dataset test.json \
    --infer_backend vllm \
    --batch_size 32 \
    --save_name results.jsonl
```

### Compute Metrics

```bash
# Text-to-SVG: CLIP score + preference metrics
python -m hivg_metric.compute_metrics \
    --input results.jsonl \
    --task_type text2svg \
    --metrics basic,clip,preference \
    --output metrics.json

# Image-to-SVG: visual similarity metrics
python -m hivg_metric.compute_metrics \
    --input results.jsonl \
    --task_type img2svg \
    --image_base_dir /path/to/images \
    --metrics basic,visual \
    --output metrics.json
```

### Supported Metrics

| Metric Group | img2svg | text2svg | Description |
|:---|:---:|:---:|:---|
| `basic` | ✓ | ✓ | Success rate, SVG length, path count |
| `visual` | ✓ | — | SSIM, LPIPS, PSNR vs. input image |
| `clip` | — | ✓ | CLIP score (text–image alignment) |
| `preference` | ✓ | ✓ | PickScore, ImageReward, HPS, Aesthetic |
| `diversity` | — | ✓ | DINOv2 / CLIP diversity (N > 1 samples) |

### Visualize Results

```bash
python -m hivg_metric.visualize \
    --input results.jsonl \
    --output report.html \
    --title "HiVG Evaluation Report"
```

Generates an interactive HTML report with side-by-side input / prediction / ground-truth columns.

---

## Dataset Format

<details>
<summary><b>Alpaca (text2svg)</b></summary>

```json
[
  {"instruction": "Draw a red circle", "input": "", "output": "<svg tokens>"}
]
```
</details>

<details>
<summary><b>ShareGPT (img2svg)</b></summary>

```json
[
  {
    "messages": [{"role": "user", "content": "<image>Convert to SVG"}],
    "images": ["path/to/image.png"],
    "label": "<svg tokens>"
  }
]
```
</details>

---

## Generation Parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| `coord_range` | 234 | Canvas coordinate range (224 canvas + margin) |
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Top-k sampling |
| `max_new_tokens` | 4096 | Maximum output length |
| `repetition_penalty` | 1.0 | Repetition penalty |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you find HiVG useful in your research, please consider citing:

```bibtex
@article{xing2026hivg,
    title={Hierarchical SVG Tokenization: Learning Compact Visual Programs for Scalable Vector Graphics Modeling},
    author={Xing, Ximing and Xue, Ziteng and Li, Zhenxi and Liang, Weicong and Wang, Linqing and Yang, Zhantao and Hang, Tiankai and Yin, Zijin and Lu, Qinglin and Wang, Chunyu and Yu, Qian},
    journal={arXiv preprint},
    year={2026}
}
```
