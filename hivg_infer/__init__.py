"""
HiSVG Inference Pipeline.

SVG generation inference pipeline built on HuggingFace transformers.
No LLaMA-Factory dependency required.

Usage:
    from hivg_infer import HiSVGInferencePipeline

    pipeline = HiSVGInferencePipeline(
        model_path="path/to/model",
        adapter_path="path/to/lora",  # optional
        coord_range=234,
    )

    # Single inference
    result = pipeline.text2svg("Draw a red circle")
    print(result["svg"])

    # Batch inference
    results = pipeline.generate_batch("test.json", output_dir="./outputs")
"""

__version__ = "0.1.0"

from .pipeline import HiSVGInferencePipeline
from .postprocess import extract_svg_tokens, tokens_to_svg

__all__ = [
    "__version__",
    "HiSVGInferencePipeline",
    "extract_svg_tokens",
    "tokens_to_svg",
]
