"""
HiVG-Tokenizer: SVG Tokenizer for Language Models.

Three-level tokenization:
- Level 0 (Raw):    Original SVG string
- Level 1 (Atomic): Discrete tokens (<cmd_M> <P_100> <P_200> <cmd_c> <d_0> <d_5> ...)
- Level 2 (BPE):    Merged segments (<cmd_M> <P_100> <P_200> <SEG_0> ...)

Command tokens use cmd_ prefix to avoid vocabulary conflicts (e.g., <s> in Qwen).

Usage:
    from hivg_tokenizer import SVGTokenizer, SVGDetokenizer
    from hivg_tokenizer import tokens_to_string, string_to_tokens

    tokenizer = SVGTokenizer(coord_range=234)
    tokens = tokenizer.tokenize(svg_string)

    # Serialize to string (no-space format saves ~19% Qwen tokens)
    token_str = tokens_to_string(tokens)  # '<svg><path><cmd_M>...'

    # Parse back to list
    tokens = string_to_tokens(token_str)

    # Reconstruct SVG
    detokenizer = SVGDetokenizer()
    svg_string = detokenizer.detokenize(tokens)
"""

__version__ = "0.1.0"

from .tokens import (
    SVGTokens,
    COMMAND_PARAMS,
    COMMAND_TOKEN_MAP,
    TOKEN_TO_COMMAND_MAP,
    STRUCTURE_OPEN,
    STRUCTURE_CLOSE,
    SELF_CLOSING,
    ARC_FLAGS,
)
from .tokenizer import SVGTokenizer, tokens_to_string, string_to_tokens
from .detokenizer import SVGDetokenizer
from .bpe import SVGBPETrainer

__all__ = [
    "__version__",
    "SVGTokens",
    "SVGTokenizer",
    "SVGDetokenizer",
    "SVGBPETrainer",
    "tokens_to_string",
    "string_to_tokens",
    "COMMAND_PARAMS",
    "COMMAND_TOKEN_MAP",
    "TOKEN_TO_COMMAND_MAP",
    "STRUCTURE_OPEN",
    "STRUCTURE_CLOSE",
    "SELF_CLOSING",
    "ARC_FLAGS",
]
