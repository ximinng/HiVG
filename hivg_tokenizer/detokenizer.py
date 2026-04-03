"""
SVG Detokenizer: token sequence -> SVG string.

Converts token sequence back to valid SVG content.
Inverse of SVGTokenizer.
"""

from typing import List, Optional, Tuple
from .tokens import (
    COMMAND_PARAMS,
    COMMAND_TOKEN_MAP,
    STRUCTURE_OPEN,
    STRUCTURE_CLOSE,
    SELF_CLOSING,
)


class SVGDetokenizer:
    """
    Convert token sequence back to SVG string.

    Token format:
    - Structure: <svg>, </svg>, <path>, </path>, ...
    - Commands: <cmd_M>, <cmd_l>, <cmd_c>, ... (using cmd_ prefix)
    - Coordinates: <P_N> (absolute), <d_N> (relative)
    - Attributes: key=value (preserved as-is)
    """

    def __init__(self, coord_range: int = 234, relative_range: int = None):
        self.coord_range = coord_range
        # Default relative_range to coord_range to match tokenizer behavior
        self.relative_range = relative_range if relative_range is not None else coord_range

    def _parse_coord_token(self, token: str) -> Optional[int]:
        """
        Parse coordinate token to integer.

        <P_100> -> 100
        <d_-50> -> -50
        """
        if token.startswith('<P_') and token.endswith('>'):
            return int(token[3:-1])
        elif token.startswith('<d_') and token.endswith('>'):
            return int(token[3:-1])
        return None

    def _is_command_token(self, token: str) -> bool:
        """Check if token is a path command."""
        return token in COMMAND_PARAMS

    def _is_structure_token(self, token: str) -> bool:
        """Check if token is a structure token."""
        return token in STRUCTURE_OPEN or token in STRUCTURE_CLOSE

    def _is_coord_token(self, token: str) -> bool:
        """Check if token is a coordinate token."""
        return (token.startswith('<P_') or token.startswith('<d_')) and token.endswith('>')

    def _is_arc_flag_token(self, token: str) -> bool:
        """Check if token is an arc flag token."""
        return token in ('<large_0>', '<large_1>', '<sweep_0>', '<sweep_1>')

    def _is_attribute_token(self, token: str) -> bool:
        """Check if token is an attribute (key=value format)."""
        return '=' in token and not token.startswith('<')

    def _reconstruct_path_d(self, tokens: List[str], start_idx: int) -> Tuple[str, int]:
        """
        Reconstruct path d attribute from tokens.

        Returns:
            (d_attribute_string, next_index)
        """
        d_parts = []
        i = start_idx
        cur_x, cur_y = 0.0, 0.0

        while i < len(tokens):
            token = tokens[i]

            # Stop at structure tokens
            if self._is_structure_token(token):
                break

            # Skip attribute tokens
            if self._is_attribute_token(token):
                i += 1
                continue

            if token == COMMAND_TOKEN_MAP['M']:
                # M command: absolute position
                if i + 2 < len(tokens):
                    x = self._parse_coord_token(tokens[i + 1])
                    y = self._parse_coord_token(tokens[i + 2])
                    if x is not None and y is not None:
                        d_parts.append(f"M {x} {y}")
                        cur_x, cur_y = float(x), float(y)
                        i += 3
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['l']:
                # l command: relative lineto
                if i + 2 < len(tokens):
                    dx = self._parse_coord_token(tokens[i + 1])
                    dy = self._parse_coord_token(tokens[i + 2])
                    if dx is not None and dy is not None:
                        d_parts.append(f"l {dx} {dy}")
                        cur_x += dx
                        cur_y += dy
                        i += 3
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['h']:
                # h command: relative horizontal lineto
                if i + 1 < len(tokens):
                    dx = self._parse_coord_token(tokens[i + 1])
                    if dx is not None:
                        d_parts.append(f"h {dx}")
                        cur_x += dx
                        i += 2
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['v']:
                # v command: relative vertical lineto
                if i + 1 < len(tokens):
                    dy = self._parse_coord_token(tokens[i + 1])
                    if dy is not None:
                        d_parts.append(f"v {dy}")
                        cur_y += dy
                        i += 2
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['c']:
                # c command: relative cubic bezier (6 coords)
                if i + 6 < len(tokens):
                    coords = []
                    valid = True
                    for j in range(6):
                        c = self._parse_coord_token(tokens[i + 1 + j])
                        if c is not None:
                            coords.append(c)
                        else:
                            valid = False
                            break
                    if valid and len(coords) == 6:
                        d_parts.append(f"c {coords[0]} {coords[1]} {coords[2]} {coords[3]} {coords[4]} {coords[5]}")
                        cur_x += coords[4]
                        cur_y += coords[5]
                        i += 7
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['s']:
                # s command: smooth cubic bezier (4 coords)
                if i + 4 < len(tokens):
                    coords = []
                    valid = True
                    for j in range(4):
                        c = self._parse_coord_token(tokens[i + 1 + j])
                        if c is not None:
                            coords.append(c)
                        else:
                            valid = False
                            break
                    if valid and len(coords) == 4:
                        d_parts.append(f"s {coords[0]} {coords[1]} {coords[2]} {coords[3]}")
                        cur_x += coords[2]
                        cur_y += coords[3]
                        i += 5
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['q']:
                # q command: relative quadratic bezier (4 coords)
                if i + 4 < len(tokens):
                    coords = []
                    valid = True
                    for j in range(4):
                        c = self._parse_coord_token(tokens[i + 1 + j])
                        if c is not None:
                            coords.append(c)
                        else:
                            valid = False
                            break
                    if valid and len(coords) == 4:
                        d_parts.append(f"q {coords[0]} {coords[1]} {coords[2]} {coords[3]}")
                        cur_x += coords[2]
                        cur_y += coords[3]
                        i += 5
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['t']:
                # t command: smooth quadratic bezier (2 coords)
                if i + 2 < len(tokens):
                    dx = self._parse_coord_token(tokens[i + 1])
                    dy = self._parse_coord_token(tokens[i + 2])
                    if dx is not None and dy is not None:
                        d_parts.append(f"t {dx} {dy}")
                        cur_x += dx
                        cur_y += dy
                        i += 3
                        continue
                i += 1

            elif token == COMMAND_TOKEN_MAP['a']:
                # a command: arc (rx, ry, rotation, large_arc, sweep, dx, dy)
                # Format: <a> rx ry rotation <large_N> <sweep_N> <d_X> <d_Y>
                if i + 7 < len(tokens):
                    try:
                        rx = float(tokens[i + 1])
                        ry = float(tokens[i + 2])
                        rotation = float(tokens[i + 3])
                        large_arc = 1 if tokens[i + 4] == '<large_1>' else 0
                        sweep = 1 if tokens[i + 5] == '<sweep_1>' else 0
                        dx = self._parse_coord_token(tokens[i + 6])
                        dy = self._parse_coord_token(tokens[i + 7])
                        if dx is not None and dy is not None:
                            d_parts.append(f"a {rx} {ry} {rotation} {large_arc} {sweep} {dx} {dy}")
                            cur_x += dx
                            cur_y += dy
                            i += 8
                            continue
                    except (ValueError, IndexError):
                        pass
                i += 1

            elif token == COMMAND_TOKEN_MAP['z'] or token == COMMAND_TOKEN_MAP.get('Z', COMMAND_TOKEN_MAP['z']):
                d_parts.append("z")
                i += 1

            else:
                # Unknown token, skip
                i += 1

        return ' '.join(d_parts), i

    def _collect_attributes(self, tokens: List[str], start_idx: int) -> Tuple[List[str], int]:
        """
        Collect attribute tokens until next structure/command token.

        Returns:
            (list of "key=value" strings, next_index)
        """
        attrs = []
        i = start_idx

        while i < len(tokens):
            token = tokens[i]
            if self._is_structure_token(token) or self._is_command_token(token):
                break
            if self._is_attribute_token(token):
                attrs.append(token)
            i += 1

        return attrs, i

    def _format_attributes(self, attrs: List[str]) -> str:
        """Format attribute list as XML attribute string."""
        if not attrs:
            return ""
        formatted = []
        for attr in attrs:
            if '=' in attr:
                key, value = attr.split('=', 1)
                # Handle special case: text content
                if key == 'text':
                    continue  # Handle separately
                formatted.append(f'{key}="{value}"')
        return ' '.join(formatted)

    def _get_text_content(self, attrs: List[str]) -> str:
        """Extract text content from attributes."""
        for attr in attrs:
            if attr.startswith('text='):
                return attr[5:]
        return ""

    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert token sequence to SVG string.

        Args:
            tokens: List of tokens

        Returns:
            SVG string
        """
        result = []
        i = 0
        indent_level = 0
        indent_str = "  "

        while i < len(tokens):
            token = tokens[i]

            if token == '<svg>':
                # Collect svg attributes (viewBox, width, height)
                svg_attrs = []
                j = i + 1
                while j < len(tokens):
                    t = tokens[j]
                    if self._is_structure_token(t) or self._is_command_token(t):
                        break
                    if self._is_attribute_token(t):
                        svg_attrs.append(t)
                    j += 1

                # Build svg tag: xmlns is predefined, others are dynamic
                attr_parts = ['xmlns="http://www.w3.org/2000/svg"']
                for attr in svg_attrs:
                    key, value = attr.split('=', 1)
                    # Skip xmlns if present in tokens (redundant)
                    if key not in ('xmlns', 'xmlns:xlink'):
                        attr_parts.append(f'{key}="{value}"')

                result.append(f'<svg {" ".join(attr_parts)}>')
                indent_level += 1
                i = j  # Skip to after svg attributes

            elif token == '</svg>':
                indent_level -= 1
                result.append('</svg>')
                i += 1

            elif token == '<defs>':
                result.append(f'{indent_str * indent_level}<defs>')
                indent_level += 1
                i += 1

            elif token == '</defs>':
                indent_level -= 1
                result.append(f'{indent_str * indent_level}</defs>')
                i += 1

            elif token == '<style>':
                # Collect CSS content
                css_content = ""
                j = i + 1
                while j < len(tokens):
                    t = tokens[j]
                    if t == '</style>':
                        break
                    # Extract CSS from css=... token
                    if t.startswith('css='):
                        css_content = t[4:]  # Remove 'css=' prefix
                    j += 1

                # Build style tag with CSS content
                result.append(f'{indent_str * indent_level}<style>')
                if css_content:
                    # Add CSS content with proper indentation
                    result.append(f'{indent_str * (indent_level + 1)}{css_content}')
                result.append(f'{indent_str * indent_level}</style>')
                i = j + 1  # Skip to after </style>

            elif token == '<filter>':
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<filter {attr_str}>')
                else:
                    result.append(f'{indent_str * indent_level}<filter>')
                indent_level += 1
                i = next_i

            elif token == '</filter>':
                indent_level -= 1
                result.append(f'{indent_str * indent_level}</filter>')
                i += 1

            # Filter effect elements (generic handling)
            elif token.startswith('<fe') and token.endswith('>'):
                elem_name = token[1:-1]
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)

                # Check if element has children
                closing_tag = f'</{elem_name}>'
                has_children = False
                j = next_i
                while j < len(tokens) and tokens[j] != closing_tag:
                    if self._is_structure_token(tokens[j]):
                        has_children = True
                        break
                    j += 1

                if has_children:
                    # Element with children
                    if attr_str:
                        result.append(f'{indent_str * indent_level}<{elem_name} {attr_str}>')
                    else:
                        result.append(f'{indent_str * indent_level}<{elem_name}>')
                    indent_level += 1
                    i = next_i
                else:
                    # Self-closing element
                    if attr_str:
                        result.append(f'{indent_str * indent_level}<{elem_name} {attr_str}/>')
                    else:
                        result.append(f'{indent_str * indent_level}<{elem_name}/>')
                    # Skip to closing tag
                    while next_i < len(tokens) and tokens[next_i] != closing_tag:
                        next_i += 1
                    i = next_i + 1

            # Closing tags for filter effects
            elif token.startswith('</fe') and token.endswith('>'):
                indent_level -= 1
                elem_name = token[2:-1]
                result.append(f'{indent_str * indent_level}</{elem_name}>')
                i += 1

            elif token == '<g>':
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<g {attr_str}>')
                else:
                    result.append(f'{indent_str * indent_level}<g>')
                indent_level += 1
                i = next_i

            elif token == '</g>':
                indent_level -= 1
                result.append(f'{indent_str * indent_level}</g>')
                i += 1

            elif token == '<path>':
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                d_attr, path_end = self._reconstruct_path_d(tokens, next_i)
                attr_str = self._format_attributes(attrs)
                if d_attr:
                    if attr_str:
                        result.append(f'{indent_str * indent_level}<path {attr_str} d="{d_attr}"/>')
                    else:
                        result.append(f'{indent_str * indent_level}<path d="{d_attr}"/>')
                else:
                    if attr_str:
                        result.append(f'{indent_str * indent_level}<path {attr_str}/>')
                    else:
                        result.append(f'{indent_str * indent_level}<path/>')
                # Skip to </path>
                while path_end < len(tokens) and tokens[path_end] != '</path>':
                    path_end += 1
                i = path_end + 1

            elif token in ('<rect>', '<circle>', '<ellipse>', '<line>', '<use>', '<image>'):
                elem_name = token[1:-1]
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<{elem_name} {attr_str}/>')
                else:
                    result.append(f'{indent_str * indent_level}<{elem_name}/>')
                # Skip to closing tag
                while next_i < len(tokens) and tokens[next_i] != f'</{elem_name}>':
                    next_i += 1
                i = next_i + 1

            elif token == '<polygon>' or token == '<polyline>':
                elem_name = token[1:-1]
                # Collect attributes and path data
                attrs = []
                j = i + 1
                while j < len(tokens) and tokens[j] != f'</{elem_name}>':
                    if self._is_attribute_token(tokens[j]):
                        attrs.append(tokens[j])
                    j += 1
                # Reconstruct points from path tokens
                points_start = i + 1
                while points_start < j and not self._is_command_token(tokens[points_start]):
                    points_start += 1
                d_attr, _ = self._reconstruct_path_d(tokens, points_start)
                # Convert path d to points
                points = self._d_to_points(d_attr)
                attr_str = self._format_attributes(attrs)
                if points:
                    if attr_str:
                        result.append(f'{indent_str * indent_level}<{elem_name} {attr_str} points="{points}"/>')
                    else:
                        result.append(f'{indent_str * indent_level}<{elem_name} points="{points}"/>')
                else:
                    if attr_str:
                        result.append(f'{indent_str * indent_level}<{elem_name} {attr_str}/>')
                    else:
                        result.append(f'{indent_str * indent_level}<{elem_name}/>')
                i = j + 1

            elif token == '<text>':
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                text_content = self._get_text_content(attrs)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<text {attr_str}>{text_content}</text>')
                else:
                    result.append(f'{indent_str * indent_level}<text>{text_content}</text>')
                # Skip to </text>
                while next_i < len(tokens) and tokens[next_i] != '</text>':
                    next_i += 1
                i = next_i + 1

            elif token == '<linearGradient>' or token == '<radialGradient>':
                grad_type = token[1:-1]
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<{grad_type} {attr_str}>')
                else:
                    result.append(f'{indent_str * indent_level}<{grad_type}>')
                indent_level += 1
                i = next_i

            elif token == '</linearGradient>' or token == '</radialGradient>':
                grad_type = token[2:-1]
                indent_level -= 1
                result.append(f'{indent_str * indent_level}</{grad_type}>')
                i += 1

            elif token == '<stop>':
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<stop {attr_str}/>')
                else:
                    result.append(f'{indent_str * indent_level}<stop/>')
                while next_i < len(tokens) and tokens[next_i] != '</stop>':
                    next_i += 1
                i = next_i + 1

            elif token in ('<clipPath>', '<mask>', '<pattern>', '<marker>', '<symbol>'):
                elem_name = token[1:-1]
                attrs, next_i = self._collect_attributes(tokens, i + 1)
                attr_str = self._format_attributes(attrs)
                if attr_str:
                    result.append(f'{indent_str * indent_level}<{elem_name} {attr_str}>')
                else:
                    result.append(f'{indent_str * indent_level}<{elem_name}>')
                indent_level += 1
                i = next_i

            elif token in ('</clipPath>', '</mask>', '</pattern>', '</marker>', '</symbol>'):
                elem_name = token[2:-1]
                indent_level -= 1
                result.append(f'{indent_str * indent_level}</{elem_name}>')
                i += 1

            else:
                # Skip unknown tokens
                i += 1

        return '\n'.join(result)

    def _d_to_points(self, d_attr: str) -> str:
        """Convert path d attribute to polygon/polyline points string."""
        if not d_attr:
            return ""

        points = []
        cur_x, cur_y = 0.0, 0.0

        # Simple parser for M and l commands
        parts = d_attr.split()
        i = 0
        while i < len(parts):
            cmd = parts[i]
            if cmd == 'M' and i + 2 < len(parts):
                cur_x = float(parts[i + 1])
                cur_y = float(parts[i + 2])
                points.append(f"{cur_x},{cur_y}")
                i += 3
            elif cmd == 'l' and i + 2 < len(parts):
                cur_x += float(parts[i + 1])
                cur_y += float(parts[i + 2])
                points.append(f"{cur_x},{cur_y}")
                i += 3
            elif cmd == 'z':
                i += 1
            else:
                i += 1

        return ' '.join(points)
