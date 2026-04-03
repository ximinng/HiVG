"""
SVG Tokenizer: SVG string -> token sequence.

Converts SVG content into a sequence of discrete tokens:
- Structure tokens: <svg>, <path>, <defs>, ...
- Command tokens: <cmd_M>, <cmd_l>, <cmd_c>, ... (using cmd_ prefix)
- Coordinate tokens: <P_N> (absolute), <d_N> (relative)
- Attributes: preserved as raw strings (fill=#ff0000, id=grad1)
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

from .tokens import COMMAND_TOKEN_MAP


class SVGTokenizer:
    """
    Tokenize SVG content into discrete tokens.

    Coordinate system:
    - First M command uses absolute position: <P_X> <P_Y>
    - Subsequent coordinates use relative offsets: <d_X> <d_Y>
    - No X/Y distinction - position determines axis

    Attributes like fill, stroke, id are preserved as raw strings.
    """

    def __init__(
        self,
        coord_range: int = 234,
        relative_range: int = None,
        scale: float = 1.0,
    ):
        """
        Args:
            coord_range: Max absolute coordinate (0 to coord_range)
            relative_range: Max relative offset (-relative_range to +relative_range).
                           If None, defaults to coord_range to avoid clipping large deltas.
            scale: Scale factor for coordinates (default 1.0)
        """
        self.coord_range = coord_range
        # Default relative_range to coord_range to handle any delta within the canvas
        self.relative_range = relative_range if relative_range is not None else coord_range
        self.scale = scale

    def _quantize(self, value: float) -> int:
        """Quantize float to int with scale."""
        return round(value * self.scale)

    def _clamp_abs(self, value: int) -> int:
        """Clamp to absolute coordinate range [0, coord_range]."""
        return max(0, min(self.coord_range, value))

    def _clamp_rel(self, value: int) -> int:
        """Clamp to relative coordinate range [-relative_range, +relative_range]."""
        return max(-self.relative_range, min(self.relative_range, value))

    def _parse_number(self, s: str) -> float:
        """Parse string to float, return 0 on failure."""
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    # =========================================================================
    # Path Tokenization
    # =========================================================================

    def _tokenize_path_d(self, d_attr: str) -> List[str]:
        """
        Tokenize path d attribute.

        Converts path commands and coordinates to tokens.
        First M is absolute (<P_N>), rest are relative (<d_N>).
        """
        if not d_attr:
            return []

        tokens = []
        pattern = r'([MmLlHhVvCcSsQqTtAaZz])|(-?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)'
        matches = re.findall(pattern, d_attr)

        cur_x, cur_y = 0.0, 0.0
        start_x, start_y = 0.0, 0.0
        current_cmd = None
        numbers: List[float] = []

        for cmd, num in matches:
            if cmd:
                # Process previous command
                if current_cmd and numbers:
                    cur_x, cur_y, start_x, start_y = self._process_path_command(
                        tokens, current_cmd, numbers, cur_x, cur_y, start_x, start_y
                    )
                elif current_cmd in ('Z', 'z'):
                    # Z command has no numbers, process it immediately
                    tokens.append(COMMAND_TOKEN_MAP['z'])
                    cur_x, cur_y = start_x, start_y
                current_cmd = cmd
                numbers = []
            elif num:
                numbers.append(float(num))

        # Process last command
        if current_cmd and numbers:
            self._process_path_command(tokens, current_cmd, numbers, cur_x, cur_y, start_x, start_y)
        elif current_cmd in ('Z', 'z'):
            tokens.append(COMMAND_TOKEN_MAP['z'])

        return tokens

    def _process_path_command(
        self,
        tokens: List[str],
        cmd: str,
        numbers: List[float],
        cur_x: float,
        cur_y: float,
        start_x: float,
        start_y: float,
    ) -> Tuple[float, float, float, float]:
        """Process a single path command and append tokens."""

        if cmd == 'M':
            if len(numbers) >= 2:
                x, y = numbers[0], numbers[1]
                qx = self._clamp_abs(self._quantize(x))
                qy = self._clamp_abs(self._quantize(y))
                tokens.append(COMMAND_TOKEN_MAP['M'])
                tokens.append(f"<P_{qx}>")
                tokens.append(f"<P_{qy}>")
                cur_x, cur_y = float(qx), float(qy)
                start_x, start_y = cur_x, cur_y
                # Additional pairs are implicit lineto
                for i in range(2, len(numbers) - 1, 2):
                    x, y = numbers[i], numbers[i + 1]
                    qdx = self._clamp_rel(self._quantize(x - cur_x))
                    qdy = self._clamp_rel(self._quantize(y - cur_y))
                    tokens.append(COMMAND_TOKEN_MAP['l'])
                    tokens.append(f"<d_{qdx}>")
                    tokens.append(f"<d_{qdy}>")
                    cur_x += qdx
                    cur_y += qdy

        elif cmd == 'm':
            if len(numbers) >= 2:
                dx, dy = numbers[0], numbers[1]
                qx = self._clamp_abs(self._quantize(cur_x + dx))
                qy = self._clamp_abs(self._quantize(cur_y + dy))
                tokens.append(COMMAND_TOKEN_MAP['M'])
                tokens.append(f"<P_{qx}>")
                tokens.append(f"<P_{qy}>")
                cur_x, cur_y = float(qx), float(qy)
                start_x, start_y = cur_x, cur_y
                for i in range(2, len(numbers) - 1, 2):
                    dx, dy = numbers[i], numbers[i + 1]
                    qdx = self._clamp_rel(self._quantize(dx))
                    qdy = self._clamp_rel(self._quantize(dy))
                    tokens.append(COMMAND_TOKEN_MAP['l'])
                    tokens.append(f"<d_{qdx}>")
                    tokens.append(f"<d_{qdy}>")
                    cur_x += qdx
                    cur_y += qdy

        elif cmd == 'L':
            for i in range(0, len(numbers) - 1, 2):
                x, y = numbers[i], numbers[i + 1]
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['l'])
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'l':
            for i in range(0, len(numbers) - 1, 2):
                dx, dy = numbers[i], numbers[i + 1]
                qdx = self._clamp_rel(self._quantize(dx))
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['l'])
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'H':
            for x in numbers:
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                tokens.append(COMMAND_TOKEN_MAP['h'])
                tokens.append(f"<d_{qdx}>")
                cur_x += qdx

        elif cmd == 'h':
            for dx in numbers:
                qdx = self._clamp_rel(self._quantize(dx))
                tokens.append(COMMAND_TOKEN_MAP['h'])
                tokens.append(f"<d_{qdx}>")
                cur_x += qdx

        elif cmd == 'V':
            for y in numbers:
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['v'])
                tokens.append(f"<d_{qdy}>")
                cur_y += qdy

        elif cmd == 'v':
            for dy in numbers:
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['v'])
                tokens.append(f"<d_{qdy}>")
                cur_y += qdy

        elif cmd == 'C':
            for i in range(0, len(numbers) - 5, 6):
                x1, y1 = numbers[i], numbers[i + 1]
                x2, y2 = numbers[i + 2], numbers[i + 3]
                x, y = numbers[i + 4], numbers[i + 5]
                qdx1 = self._clamp_rel(self._quantize(x1 - cur_x))
                qdy1 = self._clamp_rel(self._quantize(y1 - cur_y))
                qdx2 = self._clamp_rel(self._quantize(x2 - cur_x))
                qdy2 = self._clamp_rel(self._quantize(y2 - cur_y))
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['c'])
                tokens.append(f"<d_{qdx1}>")
                tokens.append(f"<d_{qdy1}>")
                tokens.append(f"<d_{qdx2}>")
                tokens.append(f"<d_{qdy2}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'c':
            for i in range(0, len(numbers) - 5, 6):
                dx1, dy1 = numbers[i], numbers[i + 1]
                dx2, dy2 = numbers[i + 2], numbers[i + 3]
                dx, dy = numbers[i + 4], numbers[i + 5]
                qdx1 = self._clamp_rel(self._quantize(dx1))
                qdy1 = self._clamp_rel(self._quantize(dy1))
                qdx2 = self._clamp_rel(self._quantize(dx2))
                qdy2 = self._clamp_rel(self._quantize(dy2))
                qdx = self._clamp_rel(self._quantize(dx))
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['c'])
                tokens.append(f"<d_{qdx1}>")
                tokens.append(f"<d_{qdy1}>")
                tokens.append(f"<d_{qdx2}>")
                tokens.append(f"<d_{qdy2}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'S':
            for i in range(0, len(numbers) - 3, 4):
                x2, y2 = numbers[i], numbers[i + 1]
                x, y = numbers[i + 2], numbers[i + 3]
                qdx2 = self._clamp_rel(self._quantize(x2 - cur_x))
                qdy2 = self._clamp_rel(self._quantize(y2 - cur_y))
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['s'])
                tokens.append(f"<d_{qdx2}>")
                tokens.append(f"<d_{qdy2}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 's':
            for i in range(0, len(numbers) - 3, 4):
                dx2, dy2 = numbers[i], numbers[i + 1]
                dx, dy = numbers[i + 2], numbers[i + 3]
                qdx2 = self._clamp_rel(self._quantize(dx2))
                qdy2 = self._clamp_rel(self._quantize(dy2))
                qdx = self._clamp_rel(self._quantize(dx))
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['s'])
                tokens.append(f"<d_{qdx2}>")
                tokens.append(f"<d_{qdy2}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'Q':
            for i in range(0, len(numbers) - 3, 4):
                x1, y1 = numbers[i], numbers[i + 1]
                x, y = numbers[i + 2], numbers[i + 3]
                qdx1 = self._clamp_rel(self._quantize(x1 - cur_x))
                qdy1 = self._clamp_rel(self._quantize(y1 - cur_y))
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['q'])
                tokens.append(f"<d_{qdx1}>")
                tokens.append(f"<d_{qdy1}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'q':
            for i in range(0, len(numbers) - 3, 4):
                dx1, dy1 = numbers[i], numbers[i + 1]
                dx, dy = numbers[i + 2], numbers[i + 3]
                qdx1 = self._clamp_rel(self._quantize(dx1))
                qdy1 = self._clamp_rel(self._quantize(dy1))
                qdx = self._clamp_rel(self._quantize(dx))
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['q'])
                tokens.append(f"<d_{qdx1}>")
                tokens.append(f"<d_{qdy1}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'T':
            for i in range(0, len(numbers) - 1, 2):
                x, y = numbers[i], numbers[i + 1]
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['t'])
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 't':
            for i in range(0, len(numbers) - 1, 2):
                dx, dy = numbers[i], numbers[i + 1]
                qdx = self._clamp_rel(self._quantize(dx))
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['t'])
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'A':
            for i in range(0, len(numbers) - 6, 7):
                rx, ry = numbers[i], numbers[i + 1]
                rotation = numbers[i + 2]
                large_arc = int(numbers[i + 3])
                sweep = int(numbers[i + 4])
                x, y = numbers[i + 5], numbers[i + 6]
                qdx = self._clamp_rel(self._quantize(x - cur_x))
                qdy = self._clamp_rel(self._quantize(y - cur_y))
                tokens.append(COMMAND_TOKEN_MAP['a'])
                tokens.append(f"{self._quantize(rx)}")
                tokens.append(f"{self._quantize(ry)}")
                tokens.append(f"{self._quantize(rotation)}")
                tokens.append(f"<large_{large_arc}>")
                tokens.append(f"<sweep_{sweep}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd == 'a':
            for i in range(0, len(numbers) - 6, 7):
                rx, ry = numbers[i], numbers[i + 1]
                rotation = numbers[i + 2]
                large_arc = int(numbers[i + 3])
                sweep = int(numbers[i + 4])
                dx, dy = numbers[i + 5], numbers[i + 6]
                qdx = self._clamp_rel(self._quantize(dx))
                qdy = self._clamp_rel(self._quantize(dy))
                tokens.append(COMMAND_TOKEN_MAP['a'])
                tokens.append(f"{self._quantize(rx)}")
                tokens.append(f"{self._quantize(ry)}")
                tokens.append(f"{self._quantize(rotation)}")
                tokens.append(f"<large_{large_arc}>")
                tokens.append(f"<sweep_{sweep}>")
                tokens.append(f"<d_{qdx}>")
                tokens.append(f"<d_{qdy}>")
                cur_x += qdx
                cur_y += qdy

        elif cmd in ('Z', 'z'):
            tokens.append(COMMAND_TOKEN_MAP['z'])
            cur_x, cur_y = start_x, start_y

        return cur_x, cur_y, start_x, start_y

    # =========================================================================
    # Attribute Extraction
    # =========================================================================

    def _extract_attributes(self, attr_str: str, attr_names: List[str]) -> List[str]:
        """Extract specific attributes as raw strings."""
        tokens = []
        for attr in attr_names:
            match = re.search(rf'{attr}=["\']([^"\']*)["\']', attr_str)
            if match:
                tokens.append(f"{attr}={match.group(1)}")
        return tokens

    def _extract_all_attributes(self, attr_str: str) -> List[str]:
        """Extract all attributes as raw strings."""
        tokens = []
        for match in re.finditer(r'([\w-]+)=["\']([^"\']*)["\']', attr_str):
            attr_name = match.group(1)
            attr_value = match.group(2)
            # Skip 'd' attribute (handled separately for paths)
            if attr_name != 'd':
                tokens.append(f"{attr_name}={attr_value}")
        return tokens

    # =========================================================================
    # Element Tokenization
    # =========================================================================

    def _tokenize_points(self, points_str: str, closed: bool = False) -> List[str]:
        """Tokenize polygon/polyline points."""
        tokens = []
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", points_str)

        if len(numbers) < 2:
            return tokens

        x0, y0 = float(numbers[0]), float(numbers[1])
        qx0 = self._clamp_abs(self._quantize(x0))
        qy0 = self._clamp_abs(self._quantize(y0))
        tokens.append(COMMAND_TOKEN_MAP['M'])
        tokens.append(f"<P_{qx0}>")
        tokens.append(f"<P_{qy0}>")

        cur_x, cur_y = float(qx0), float(qy0)
        for i in range(2, len(numbers) - 1, 2):
            x, y = float(numbers[i]), float(numbers[i + 1])
            qdx = self._clamp_rel(self._quantize(x - cur_x))
            qdy = self._clamp_rel(self._quantize(y - cur_y))
            tokens.append(COMMAND_TOKEN_MAP['l'])
            tokens.append(f"<d_{qdx}>")
            tokens.append(f"<d_{qdy}>")
            cur_x += qdx
            cur_y += qdy

        if closed:
            tokens.append(COMMAND_TOKEN_MAP['z'])

        return tokens

    def _tokenize_gradient(self, grad_match: re.Match, grad_type: str) -> List[str]:
        """Tokenize gradient element."""
        tokens = []
        attrs_str = grad_match.group(1)
        stops_content = grad_match.group(2)

        tokens.append(f"<{grad_type}>")
        tokens.extend(self._extract_all_attributes(attrs_str))

        for stop_match in re.finditer(r'<stop\s+([^/>]+)/?>', stops_content):
            tokens.append("<stop>")
            tokens.extend(self._extract_all_attributes(stop_match.group(1)))
            tokens.append("</stop>")

        tokens.append(f"</{grad_type}>")
        return tokens

    # =========================================================================
    # Main Tokenize Method
    # =========================================================================

    # Attributes to preserve for svg tag (dynamic, need tokenization)
    SVG_DYNAMIC_ATTRS = {'viewBox', 'width', 'height'}

    # Attributes to skip (will be hardcoded in detokenizer)
    SVG_PREDEFINED_ATTRS = {'xmlns', 'xmlns:xlink', 'enable-background', 'version'}

    def tokenize(self, svg_content: str) -> List[str]:
        """
        Tokenize complete SVG content using XML parsing.

        Args:
            svg_content: SVG string

        Returns:
            List of tokens
        """
        tokens = []

        try:
            # Parse SVG as XML
            root = ET.fromstring(svg_content)
        except ET.ParseError:
            # Fallback to regex-based parsing if XML parsing fails
            return self._tokenize_fallback(svg_content)

        # Tokenize from root
        self._tokenize_element(root, tokens)

        return tokens

    def _tokenize_element(self, elem: ET.Element, tokens: List[str]) -> None:
        """Recursively tokenize an XML element."""
        # Get tag name without namespace
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == 'svg':
            tokens.append('<svg>')
            # Extract dynamic attributes (spaces preserved as-is)
            for attr in self.SVG_DYNAMIC_ATTRS:
                if attr in elem.attrib:
                    value = elem.attrib[attr]
                    tokens.append(f"{attr}={value}")
            # Process children
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</svg>')

        elif tag == 'defs':
            tokens.append('<defs>')
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</defs>')

        elif tag == 'g':
            tokens.append('<g>')
            self._add_element_attributes(elem, tokens, exclude={'d'})
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</g>')

        elif tag == 'path':
            tokens.append('<path>')
            self._add_element_attributes(elem, tokens, exclude={'d'})
            if 'd' in elem.attrib:
                tokens.extend(self._tokenize_path_d(elem.attrib['d']))
            tokens.append('</path>')

        elif tag == 'rect':
            tokens.append('<rect>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</rect>')

        elif tag == 'circle':
            tokens.append('<circle>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</circle>')

        elif tag == 'ellipse':
            tokens.append('<ellipse>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</ellipse>')

        elif tag == 'line':
            tokens.append('<line>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</line>')

        elif tag == 'polygon':
            tokens.append('<polygon>')
            self._add_element_attributes(elem, tokens, exclude={'points'})
            if 'points' in elem.attrib:
                tokens.extend(self._tokenize_points(elem.attrib['points'], closed=True))
            tokens.append('</polygon>')

        elif tag == 'polyline':
            tokens.append('<polyline>')
            self._add_element_attributes(elem, tokens, exclude={'points'})
            if 'points' in elem.attrib:
                tokens.extend(self._tokenize_points(elem.attrib['points'], closed=False))
            tokens.append('</polyline>')

        elif tag == 'text':
            tokens.append('<text>')
            self._add_element_attributes(elem, tokens)
            if elem.text and elem.text.strip():
                tokens.append(f"text={elem.text.strip()}")
            tokens.append('</text>')

        elif tag == 'use':
            tokens.append('<use>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</use>')

        elif tag == 'linearGradient':
            tokens.append('<linearGradient>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</linearGradient>')

        elif tag == 'radialGradient':
            tokens.append('<radialGradient>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</radialGradient>')

        elif tag == 'stop':
            tokens.append('<stop>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</stop>')

        elif tag == 'clipPath':
            tokens.append('<clipPath>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</clipPath>')

        elif tag == 'mask':
            tokens.append('<mask>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</mask>')

        elif tag == 'pattern':
            tokens.append('<pattern>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</pattern>')

        elif tag == 'marker':
            tokens.append('<marker>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</marker>')

        elif tag == 'symbol':
            tokens.append('<symbol>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</symbol>')

        elif tag == 'style':
            tokens.append('<style>')
            # Preserve CSS content as a single token
            if elem.text and elem.text.strip():
                # Store CSS content with css= prefix
                tokens.append(f"css={elem.text.strip()}")
            tokens.append('</style>')

        elif tag == 'filter':
            tokens.append('<filter>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append('</filter>')

        elif tag == 'image':
            tokens.append('<image>')
            self._add_element_attributes(elem, tokens)
            tokens.append('</image>')

        # Filter effect elements (self-closing or with children)
        elif tag in ('feGaussianBlur', 'feOffset', 'feBlend', 'feColorMatrix', 'feComposite', 'feMerge'):
            tokens.append(f'<{tag}>')
            self._add_element_attributes(elem, tokens)
            for child in elem:
                self._tokenize_element(child, tokens)
            tokens.append(f'</{tag}>')

        # Skip unknown elements but process children
        else:
            for child in elem:
                self._tokenize_element(child, tokens)

    def _add_element_attributes(
        self,
        elem: ET.Element,
        tokens: List[str],
        exclude: set = None
    ) -> None:
        """Add element attributes as tokens."""
        exclude = exclude or set()
        for attr, value in elem.attrib.items():
            # Remove namespace prefix
            attr_name = attr.split('}')[-1] if '}' in attr else attr
            if attr_name not in exclude:
                tokens.append(f"{attr_name}={value}")

    def _tokenize_fallback(self, svg_content: str) -> List[str]:
        """Fallback regex-based tokenization for malformed SVG."""
        tokens = []
        tokens.append("<svg>")

        # Extract svg attributes
        svg_tag_match = re.search(r'<svg\s+([^>]*)>', svg_content)
        if svg_tag_match:
            svg_attrs = svg_tag_match.group(1)
            for attr in self.SVG_DYNAMIC_ATTRS:
                match = re.search(rf'{attr}=["\']([^"\']*)["\']', svg_attrs)
                if match:
                    tokens.append(f"{attr}={match.group(1)}")

        # Simple path extraction
        for match in re.finditer(r'<path\s+([^>]*)/?>', svg_content):
            attr_str = match.group(1)
            tokens.append("<path>")
            # Extract attributes
            for attr_match in re.finditer(r'([\w-]+)=["\']([^"\']*)["\']', attr_str):
                if attr_match.group(1) != 'd':
                    tokens.append(f"{attr_match.group(1)}={attr_match.group(2)}")
            # Extract d attribute
            d_match = re.search(r'd=["\']([^"\']*)["\']', attr_str)
            if d_match:
                tokens.extend(self._tokenize_path_d(d_match.group(1)))
            tokens.append("</path>")

        tokens.append("</svg>")
        return tokens


# =============================================================================
# Token String Serialization (Smart Space Format)
# =============================================================================

# Pattern to detect start of a new attribute: letter followed by word chars and =
_ATTR_START_PATTERN = re.compile(r'[a-zA-Z][\w-]*=')


def _is_attr_token(token: str) -> bool:
    """Check if token is an attribute (key=value format)."""
    return '=' in token and not token.startswith('<')


def _is_bare_number(token: str) -> bool:
    """Check if token is a bare number (not wrapped in <> or containing =)."""
    if not token or token.startswith('<') or '=' in token:
        return False
    # Check if it looks like a number: optional sign, digits, optional decimal
    # Supports: 123, -45, 0.5, -.5, .5, -0.5
    s = token.strip()
    if not s:
        return False
    i = 0
    if s[i] in '+-':
        i += 1
    if i >= len(s):
        return False
    has_digit = False
    # Integer part
    while i < len(s) and s[i].isdigit():
        has_digit = True
        i += 1
    # Decimal part
    if i < len(s) and s[i] == '.':
        i += 1
        while i < len(s) and s[i].isdigit():
            has_digit = True
            i += 1
    return has_digit and i == len(s)


def tokens_to_string(tokens: List[str], use_space: bool = False) -> str:
    """
    Convert token list to string.

    Args:
        tokens: List of SVG tokens
        use_space: If True, use full space separator (legacy format).
                   If False (default), use smart spacing (saves ~19% Qwen tokens).
                   Smart spacing only adds space between consecutive attributes
                   to avoid ambiguity (e.g., 'fill=#fffstroke=...' is ambiguous).

    Returns:
        Token string

    Example:
        # Full space: '<svg> viewBox=0 0 1008 1008 <path> fill=#fff <cmd_M> <P_100>'
        # Smart space: '<svg>viewBox=0 0 1008 1008<path>fill=#fff opacity=0.5<cmd_M><P_100>'
        # Note: space kept between consecutive attrs (fill=#fff opacity=0.5)
    """
    if use_space:
        return ' '.join(tokens)

    # Smart join: add space between tokens that would be ambiguous without it
    if not tokens:
        return ''

    result = [tokens[0]]
    for i in range(1, len(tokens)):
        prev, curr = tokens[i - 1], tokens[i]
        # Add space if:
        # 1. Both are attributes (e.g., fill=#fff stroke=#000)
        # 2. Both are bare numbers (e.g., 50 30 45 for arc rx ry rotation)
        # 3. Bare number followed by attribute or vice versa
        prev_is_attr = _is_attr_token(prev)
        prev_is_num = _is_bare_number(prev)
        curr_is_attr = _is_attr_token(curr)
        curr_is_num = _is_bare_number(curr)

        need_space = (
            (prev_is_attr and curr_is_attr) or
            (prev_is_num and curr_is_num) or
            (prev_is_num and curr_is_attr) or
            (prev_is_attr and curr_is_num)
        )
        if need_space:
            result.append(' ')
        result.append(curr)

    return ''.join(result)


def string_to_tokens(token_string: str) -> List[str]:
    """
    Parse token string back to token list.

    Supports Smart Space format where:
    - Angle bracket tokens (<svg>, <cmd_M>, etc.) have no spaces between them
    - Attribute values can contain spaces (e.g., viewBox=0 0 1008 1008)
    - Consecutive attributes are space-separated

    Args:
        token_string: Token string in smart-space format

    Returns:
        List of tokens

    Example:
        '<svg>viewBox=0 0 1008 1008<path>fill=#fff opacity=0.5<cmd_M>'
        -> ['<svg>', 'viewBox=0 0 1008 1008', '<path>', 'fill=#fff', 'opacity=0.5', '<cmd_M>']
    """
    if not token_string:
        return []

    tokens = []
    i = 0
    n = len(token_string)

    while i < n:
        char = token_string[i]

        if char == '<':
            # Angle bracket token: read until '>'
            j = token_string.find('>', i)
            if j == -1:
                j = n - 1
            tokens.append(token_string[i:j + 1])
            i = j + 1

        elif char.isalpha():
            # Potential attribute: name=value
            # Find the '=' first
            eq_pos = token_string.find('=', i)
            if eq_pos == -1 or token_string.find('<', i, eq_pos) != -1:
                # No '=' before next '<', skip this char
                i += 1
                continue

            # Find where this attribute value ends:
            # - At next '<'
            # - At next attribute pattern (space + letter...=)
            value_start = eq_pos + 1
            value_end = n

            # Look for next '<'
            next_angle = token_string.find('<', value_start)
            if next_angle != -1:
                value_end = min(value_end, next_angle)

            # Look for next attribute pattern within the range
            search_region = token_string[value_start:value_end]
            # Find pattern: space followed by attr=
            for match in _ATTR_START_PATTERN.finditer(search_region):
                # Check if there's a space before this match
                pos = match.start()
                if pos > 0 and search_region[pos - 1] == ' ':
                    value_end = value_start + pos - 1  # Don't include the space
                    break

            tokens.append(token_string[i:value_end].rstrip())
            i = value_end
            # Skip leading space before next token
            while i < n and token_string[i] == ' ':
                i += 1

        elif char == ' ':
            # Skip standalone spaces
            i += 1

        elif char in '0123456789+-.' :
            # Bare number token (e.g., arc rx/ry/rotation)
            # Read until space, '<', or start of attribute
            j = i
            # Handle optional sign
            if j < n and token_string[j] in '+-':
                j += 1
            # Read digits before decimal
            while j < n and token_string[j].isdigit():
                j += 1
            # Read decimal part
            if j < n and token_string[j] == '.':
                j += 1
                while j < n and token_string[j].isdigit():
                    j += 1
            # Only add if we actually read a number
            if j > i and _is_bare_number(token_string[i:j]):
                tokens.append(token_string[i:j])
                i = j
            else:
                i += 1

        else:
            # Unknown char, skip
            i += 1

    return tokens
