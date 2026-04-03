"""
SVG Token definitions.

Token categories:
1. Structure tokens - SVG elements (<svg>, <path>, <defs>, ...)
2. Command tokens - Path commands (<cmd_M>, <cmd_l>, <cmd_c>, ...)
3. Coordinate tokens - Position values (<P_N>, <d_N>)
4. Arc flag tokens - Arc parameters (<large_0>, <sweep_1>)

Token definitions are loaded from configs/tokens.yaml
Coordinate tokens are generated dynamically based on coord_range.

Command tokens use cmd_ prefix to avoid conflicts with LLM special tokens
(e.g., <s> exists in Qwen as BOS token).
"""

import yaml
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field


# =============================================================================
# Load Token Config
# =============================================================================

CONFIG_PATH = Path(__file__).parent / "configs" / "tokens.yaml"


def _load_token_config() -> dict:
    """Load token definitions from YAML config."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


# Load config at module import
_TOKEN_CONFIG = _load_token_config()


# =============================================================================
# Build COMMAND_PARAMS from config
# =============================================================================

def _build_command_params() -> Dict[str, int]:
    """
    Build COMMAND_PARAMS dict from config.

    Returns dict mapping command token to param count:
    {"<cmd_M>": 2, "<cmd_m>": 2, "<cmd_L>": 2, ...}
    """
    params = {}
    commands = _TOKEN_CONFIG.get("command", {}).get("commands", {})

    for cmd_info in commands.values():
        tokens = cmd_info.get("tokens", [])
        param_count = cmd_info.get("params", 0)
        for token in tokens:
            params[f"<{token}>"] = param_count

    return params


COMMAND_PARAMS: Dict[str, int] = _build_command_params()


def _build_command_token_map() -> Dict[str, str]:
    """
    Build mapping from SVG path command to token string.

    Returns dict mapping SVG command to token:
    {"M": "<cmd_M>", "m": "<cmd_m>", "L": "<cmd_L>", ...}

    This allows tokenizer/detokenizer to use config-driven tokens
    instead of hardcoding.
    """
    cmd_map = {}
    commands = _TOKEN_CONFIG.get("command", {}).get("commands", {})

    for cmd_info in commands.values():
        tokens = cmd_info.get("tokens", [])
        for token in tokens:
            # Extract the SVG command letter from token name
            # e.g., "cmd_M" -> "M", "cmd_m" -> "m"
            if token.startswith("cmd_"):
                svg_cmd = token[4:]  # Remove "cmd_" prefix
                cmd_map[svg_cmd] = f"<{token}>"

    return cmd_map


COMMAND_TOKEN_MAP: Dict[str, str] = _build_command_token_map()


def _build_token_to_command_map() -> Dict[str, str]:
    """
    Build reverse mapping from token string to SVG path command.

    Returns dict mapping token to SVG command:
    {"<cmd_M>": "M", "<cmd_m>": "m", "<cmd_L>": "L", ...}
    """
    return {v: k for k, v in COMMAND_TOKEN_MAP.items()}


TOKEN_TO_COMMAND_MAP: Dict[str, str] = _build_token_to_command_map()


# =============================================================================
# Build Structure Token Sets from config
# =============================================================================

def _build_structure_tokens() -> tuple:
    """
    Build STRUCTURE_OPEN and STRUCTURE_CLOSE sets from config.

    Returns:
        (STRUCTURE_OPEN, STRUCTURE_CLOSE, SELF_CLOSING)
    """
    elements = _TOKEN_CONFIG.get("structure", {}).get("elements", {})
    open_tokens = {f"<{e}>" for e in elements}
    close_tokens = {f"</{e}>" for e in elements}
    # Self-closing elements (no children in SVG)
    self_closing = {'<rect>', '<circle>', '<ellipse>', '<line>', '<use>', '<stop>'}
    return open_tokens, close_tokens, self_closing


STRUCTURE_OPEN, STRUCTURE_CLOSE, SELF_CLOSING = _build_structure_tokens()


# =============================================================================
# Build Arc Flag Token Set from config
# =============================================================================

def _build_arc_flags() -> set:
    """
    Build ARC_FLAGS set from config.

    Returns set of arc flag tokens:
    {"<large_0>", "<large_1>", "<sweep_0>", "<sweep_1>"}
    """
    flags = _TOKEN_CONFIG.get("arc_flag", {}).get("flags", {})
    return {f"<{flag}>" for flag in flags.keys()}


ARC_FLAGS: set = _build_arc_flags()


# =============================================================================
# SVGTokens Class
# =============================================================================

@dataclass
class SVGTokens:
    """
    SVG token vocabulary builder.

    Coordinate system:
    - Absolute position: <P_N> where N in [0, coord_range]
    - Relative offset: <d_N> where N in [-relative_range, relative_range]

    No X/Y distinction - position in sequence determines axis.

    Token definitions are loaded from configs/tokens.yaml
    """

    coord_range: int = 234
    relative_range: int = 234
    _config: dict = field(default_factory=lambda: _TOKEN_CONFIG, repr=False)

    def get_structure_tokens(self) -> List[str]:
        """SVG element tokens (loaded from config)."""
        tokens = []
        elements = self._config.get("structure", {}).get("elements", {})

        for elem_name in elements.keys():
            tokens.append(f"<{elem_name}>")
            tokens.append(f"</{elem_name}>")

        return tokens

    def get_command_tokens(self) -> List[str]:
        """Path command tokens (loaded from config)."""
        return list(COMMAND_PARAMS.keys())

    def get_coordinate_tokens(self) -> List[str]:
        """
        Coordinate tokens (generated dynamically).

        - <P_N>: Absolute position (0 to coord_range)
        - <d_N>: Relative offset (-relative_range to +relative_range)
        """
        tokens = []

        # Absolute position: P_0 to P_coord_range
        for i in range(self.coord_range + 1):
            tokens.append(f"<P_{i}>")

        # Relative offset: d_-N to d_+N
        for i in range(-self.relative_range, self.relative_range + 1):
            tokens.append(f"<d_{i}>")

        return tokens

    def get_arc_flag_tokens(self) -> List[str]:
        """Arc flag tokens (loaded from config)."""
        flags = self._config.get("arc_flag", {}).get("flags", {})
        return [f"<{flag}>" for flag in flags.keys()]

    def get_all_tokens(self) -> List[str]:
        """Get all SVG tokens."""
        tokens = []
        tokens.extend(self.get_structure_tokens())
        tokens.extend(self.get_command_tokens())
        tokens.extend(self.get_coordinate_tokens())
        tokens.extend(self.get_arc_flag_tokens())
        return tokens

    def get_stats(self) -> Dict[str, int]:
        """Get token statistics."""
        return {
            "structure": len(self.get_structure_tokens()),
            "commands": len(self.get_command_tokens()),
            "coordinates_P": self.coord_range + 1,
            "coordinates_d": 2 * self.relative_range + 1,
            "coordinates_total": (self.coord_range + 1) + (2 * self.relative_range + 1),
            "arc_flags": len(self.get_arc_flag_tokens()),
            "total": len(self.get_all_tokens()),
        }

    def print_stats(self):
        """Print token statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("SVG TOKEN STATISTICS")
        print("=" * 50)
        print(f"{'Structure tokens':<30} {stats['structure']:>8}")
        print(f"{'Command tokens':<30} {stats['commands']:>8}")
        print(f"{'Absolute P (0~{})'.format(self.coord_range):<30} {stats['coordinates_P']:>8}")
        print(f"{'Relative d (-{}~+{})'.format(self.relative_range, self.relative_range):<30} {stats['coordinates_d']:>8}")
        print(f"{'Arc flags':<30} {stats['arc_flags']:>8}")
        print("-" * 50)
        print(f"{'TOTAL':<30} {stats['total']:>8}")


# =============================================================================
# Convenience Functions
# =============================================================================

def build_svg_tokens(coord_range: int = 234, relative_range: int = 234) -> List[str]:
    """Build complete SVG token list."""
    return SVGTokens(coord_range, relative_range).get_all_tokens()


def get_command_param_count(cmd: str) -> int:
    """Get parameter count for a path command token."""
    return COMMAND_PARAMS.get(cmd, 0)


def reload_config():
    """Reload token config from YAML file."""
    global _TOKEN_CONFIG, COMMAND_PARAMS, COMMAND_TOKEN_MAP, TOKEN_TO_COMMAND_MAP
    global STRUCTURE_OPEN, STRUCTURE_CLOSE, SELF_CLOSING, ARC_FLAGS
    _TOKEN_CONFIG = _load_token_config()
    COMMAND_PARAMS = _build_command_params()
    COMMAND_TOKEN_MAP = _build_command_token_map()
    TOKEN_TO_COMMAND_MAP = _build_token_to_command_map()
    STRUCTURE_OPEN, STRUCTURE_CLOSE, SELF_CLOSING = _build_structure_tokens()
    ARC_FLAGS = _build_arc_flags()
