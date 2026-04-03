"""
SVG token postprocessing: model output -> final SVG.
"""

import re
import xml.etree.ElementTree as ET
from typing import Optional

from hivg_tokenizer import SVGDetokenizer, string_to_tokens


def extract_svg_tokens(model_output: str) -> str:
    """
    Extract SVG token sequence from model output.

    Model may output additional text before/after SVG tokens.
    This function extracts the <svg>...</svg> token sequence.

    Args:
        model_output: Raw model output string

    Returns:
        SVG token string (from <svg> to </svg>)
    """
    # Find <svg> ... </svg> pattern
    # Handle both with and without spaces between tokens
    pattern = r'<svg>.*?</svg>'
    match = re.search(pattern, model_output, re.DOTALL)

    if match:
        return match.group(0)

    # Fallback: if no </svg> found, take from <svg> to end
    svg_start = model_output.find('<svg>')
    if svg_start != -1:
        return model_output[svg_start:]

    # No SVG tokens found, return as-is
    return model_output


def tokens_to_svg(
    token_str: str,
    coord_range: int = 234,
    relative_range: Optional[int] = None,
) -> str:
    """
    Convert SVG token string to final SVG code.

    Args:
        token_str: SVG token string (e.g., '<svg><path><cmd_M><P_100>...')
        coord_range: Coordinate range for detokenizer
        relative_range: Relative coordinate range (defaults to coord_range)

    Returns:
        Valid SVG string
    """
    # Parse token string to list
    tokens = string_to_tokens(token_str)

    # Detokenize to SVG
    detokenizer = SVGDetokenizer(
        coord_range=coord_range,
        relative_range=relative_range,
    )
    svg = detokenizer.detokenize(tokens)

    return svg


def postprocess_output(
    model_output: str,
    coord_range: int = 234,
    relative_range: Optional[int] = None,
) -> dict:
    """
    Full postprocessing pipeline: model output -> SVG + metadata.

    Args:
        model_output: Raw model output string
        coord_range: Coordinate range for detokenizer
        relative_range: Relative coordinate range

    Returns:
        dict with keys:
            - svg: Final SVG string
            - tokens: Extracted token string
            - raw_output: Original model output
            - success: Whether conversion succeeded
            - error: Error message if failed
    """
    result = {
        "raw_output": model_output,
        "tokens": None,
        "svg": None,
        "success": False,
        "error": None,
    }

    try:
        # Extract SVG tokens
        token_str = extract_svg_tokens(model_output)
        result["tokens"] = token_str

        # Convert to SVG
        svg = tokens_to_svg(token_str, coord_range, relative_range)
        result["svg"] = svg

        # Validate SVG XML so success aligns better with downstream renderability.
        svg_lower = svg.lower().strip()
        if "<svg" not in svg_lower:
            raise ValueError("Invalid SVG: missing <svg> tag")
        if "</svg>" not in svg_lower:
            raise ValueError("Invalid SVG: missing </svg> closing tag")
        ET.fromstring(svg)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
