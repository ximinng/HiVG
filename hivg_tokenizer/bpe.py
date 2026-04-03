"""
Semantic BPE for SVG tokens.

Key difference from standard BPE:
- Only merges COMPLETE path command segments
- A complete segment = command token + all required coordinate tokens
- Prevents learning partial/meaningless patterns like <cmd_c> <d_0> (incomplete cubic)

Example complete segments:
- <cmd_l> <d_X> <d_Y>                                    (lineto: 2 coords)
- <cmd_c> <d_X1> <d_Y1> <d_X2> <d_Y2> <d_X> <d_Y>       (cubic: 6 coords)
- <cmd_q> <d_X1> <d_Y1> <d_X> <d_Y>                     (quadratic: 4 coords)
"""

import heapq
import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .tokens import COMMAND_PARAMS, STRUCTURE_OPEN, STRUCTURE_CLOSE, COMMAND_TOKEN_MAP, ARC_FLAGS

# Pre-computed token sets for performance
_Z_TOKENS = {COMMAND_TOKEN_MAP.get('Z'), COMMAND_TOKEN_MAP.get('z')}
_ARC_TOKENS = {COMMAND_TOKEN_MAP.get('A'), COMMAND_TOKEN_MAP.get('a')}


@dataclass
class BPESegment:
    """A learned BPE segment (complete command sequence)."""
    id: int
    tokens: Tuple[str, ...]
    frequency: int

    @property
    def token_name(self) -> str:
        return f"<SEG_{self.id}>"


class SVGBPETrainer:
    """
    Semantic BPE trainer for SVG tokens.

    Only learns merges of complete path command segments.
    """

    # Structure tokens act as segment boundaries (from config)
    STRUCTURE_TOKENS = STRUCTURE_OPEN | STRUCTURE_CLOSE

    def __init__(
        self,
        coord_range: int = 234,
        relative_range: Optional[int] = None,
        min_frequency: int = 5,
    ):
        self.coord_range = coord_range
        # Match the L1 tokenizer default: relative offsets span [-coord_range, +coord_range].
        self.relative_range = relative_range if relative_range is not None else coord_range
        self.min_frequency = min_frequency

        # Learned data
        self.merges: List[Tuple[Tuple[str, ...], str]] = []
        self.segments: List[BPESegment] = []
        self._encode_index: Optional[Dict[str, List[Tuple[Tuple[str, ...], int]]]] = None

    def _is_command_token(self, token: str) -> bool:
        """Check if token is a path command."""
        return token in COMMAND_PARAMS

    def _is_coord_token(self, token: str) -> bool:
        """Check if token is a coordinate token."""
        return (token.startswith('<P_') or token.startswith('<d_')) and token.endswith('>')

    def _is_arc_flag_token(self, token: str) -> bool:
        """Check if token is an arc flag."""
        return token in ARC_FLAGS

    def _is_arc_number(self, token: str) -> bool:
        """Check if token is an arc numeric parameter (rx, ry, rotation)."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _get_command_param_count(self, cmd: str) -> int:
        """Get number of coordinate parameters for a command."""
        return COMMAND_PARAMS.get(cmd, 0)

    def _build_encode_index(self) -> Dict[str, List[Tuple[Tuple[str, ...], int]]]:
        """Build first-token index for fast encode. Cached after first call."""
        if self._encode_index is not None:
            return self._encode_index

        from collections import defaultdict
        index = defaultdict(list)
        for seg in self.segments:
            first_token = seg.tokens[0]
            index[first_token].append((seg.tokens, seg.id))

        # Sort each bucket by length descending (greedy longest-first)
        for key in index:
            index[key].sort(key=lambda x: len(x[0]), reverse=True)

        self._encode_index = dict(index)
        return self._encode_index

    # =========================================================================
    # Extract Complete Command Segments
    # =========================================================================

    def _extract_complete_segments(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Extract complete path command segments from token sequence.

        A complete segment is:
        - Command token + all required coordinate tokens
        - Example: <cmd_c> + 6 coords = complete cubic bezier

        Returns:
            List of complete segment tuples
        """
        segments = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Skip structure tokens and attributes
            if token in self.STRUCTURE_TOKENS:
                i += 1
                continue

            if '=' in token and not token.startswith('<'):
                i += 1
                continue

            # Check if this is a command token
            if self._is_command_token(token):
                segment, consumed = self._try_extract_segment(tokens, i)
                if segment:
                    segments.append(segment)
                    i += consumed
                else:
                    i += 1
            else:
                i += 1

        return segments

    def _try_extract_segment(self, tokens: List[str], start: int) -> Tuple[Optional[Tuple[str, ...]], int]:
        """
        Try to extract a complete command segment starting at index.

        Returns:
            (segment_tuple or None, number_of_tokens_consumed)
        """
        cmd = tokens[start]
        param_count = self._get_command_param_count(cmd)

        # Z command has no parameters
        if cmd in _Z_TOKENS:
            return (cmd,), 1

        # Arc command is special: rx ry rotation <large_N> <sweep_N> dx dy
        if cmd in _ARC_TOKENS:
            return self._try_extract_arc_segment(tokens, start)

        # Other commands: command + N coordinate tokens
        if param_count == 0:
            return (cmd,), 1

        segment = [cmd]
        i = start + 1
        coords_found = 0

        while i < len(tokens) and coords_found < param_count:
            t = tokens[i]

            # Stop at structure tokens or new commands
            if t in self.STRUCTURE_TOKENS or self._is_command_token(t):
                break

            # Skip attribute tokens
            if '=' in t and not t.startswith('<'):
                i += 1
                continue

            # Collect coordinate token
            if self._is_coord_token(t):
                segment.append(t)
                coords_found += 1
                i += 1
            else:
                # Unknown token, stop
                break

        # Only return if we got all required coordinates
        if coords_found == param_count:
            return tuple(segment), len(segment)
        else:
            return None, 1

    def _try_extract_arc_segment(self, tokens: List[str], start: int) -> Tuple[Optional[Tuple[str, ...]], int]:
        """
        Extract arc command segment.

        Format: <a> rx ry rotation <large_N> <sweep_N> <d_X> <d_Y>
        """
        cmd = tokens[start]
        segment = [cmd]
        i = start + 1

        # Expect: rx, ry, rotation (3 numbers)
        nums_found = 0
        while i < len(tokens) and nums_found < 3:
            t = tokens[i]
            if self._is_arc_number(t):
                segment.append(t)
                nums_found += 1
                i += 1
            elif '=' in t and not t.startswith('<'):
                i += 1  # Skip attributes
            else:
                break

        if nums_found < 3:
            return None, 1

        # Expect: <large_N>
        if i < len(tokens) and self._is_arc_flag_token(tokens[i]):
            segment.append(tokens[i])
            i += 1
        else:
            return None, 1

        # Expect: <sweep_N>
        if i < len(tokens) and self._is_arc_flag_token(tokens[i]):
            segment.append(tokens[i])
            i += 1
        else:
            return None, 1

        # Expect: 2 coordinate tokens (dx, dy)
        coords_found = 0
        while i < len(tokens) and coords_found < 2:
            t = tokens[i]
            if self._is_coord_token(t):
                segment.append(t)
                coords_found += 1
                i += 1
            elif '=' in t and not t.startswith('<'):
                i += 1
            else:
                break

        if coords_found == 2:
            return tuple(segment), len(segment)
        else:
            return None, 1

    # =========================================================================
    # BPE Training
    # =========================================================================

    def _count_segment_pairs(self, segments: List[Tuple[str, ...]]) -> Counter:
        """
        Count adjacent segment pairs.

        Only counts pairs of COMPLETE segments.
        """
        pair_counts = Counter()
        for i in range(len(segments) - 1):
            pair = (segments[i], segments[i + 1])
            pair_counts[pair] += 1
        return pair_counts

    def _merge_segment_pair(
        self,
        segments: List[Tuple[str, ...]],
        pair: Tuple[Tuple[str, ...], Tuple[str, ...]],
        new_segment: Tuple[str, ...]
    ) -> List[Tuple[str, ...]]:
        """Replace all occurrences of pair with merged segment."""
        result = []
        i = 0
        while i < len(segments):
            if i < len(segments) - 1 and segments[i] == pair[0] and segments[i + 1] == pair[1]:
                result.append(new_segment)
                i += 2
            else:
                result.append(segments[i])
                i += 1
        return result

    def train(
        self,
        token_sequences: Iterable[List[str]],
        num_merges: int = 500,
        verbose: bool = True
    ) -> List[BPESegment]:
        """
        Train BPE on token sequences.

        Optimized: integer IDs + incremental pair count updates.
        """
        if verbose:
            print("Extracting complete segments...")

        # === Integer ID mapping for fast hash/compare ===
        seg_to_id: Dict[Tuple[str, ...], int] = {}
        id_to_seg: Dict[int, Tuple[str, ...]] = {}

        def get_id(seg: Tuple[str, ...]) -> int:
            if seg not in seg_to_id:
                sid = len(seg_to_id)
                seg_to_id[seg] = sid
                id_to_seg[sid] = seg
            return seg_to_id[seg]

        int_seqs = []
        total_segments = 0
        total_sequences = 0
        for tokens in token_sequences:
            segs = self._extract_complete_segments(tokens)
            int_seq = [get_id(seg) for seg in segs]
            int_seqs.append(int_seq)
            total_segments += len(int_seq)
            total_sequences += 1

        if verbose:
            print(
                f"Extracted {total_segments} complete segments "
                f"from {total_sequences} sequences"
            )
            print(f"Starting BPE training (max {num_merges} merges, min_freq={self.min_frequency})...")
            print(f"Unique segments: {len(seg_to_id)}")

        # === Build initial pair counts + pair→sequence index (ONE pass) ===
        pair_counts: Dict[Tuple[int, int], int] = {}
        pair_to_seqs: Dict[Tuple[int, int], Set[int]] = {}
        for seq_idx, seq in enumerate(int_seqs):
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + 1
                if p not in pair_to_seqs:
                    pair_to_seqs[p] = set()
                pair_to_seqs[p].add(seq_idx)

        if verbose:
            print(f"Unique pairs: {len(pair_counts)}")

        # === Build lazy-delete max-heap ===
        heap = [(-cnt, p) for p, cnt in pair_counts.items()]
        heapq.heapify(heap)

        # === Merge loop: heap + pair_to_seqs ===
        for merge_idx in range(num_merges):
            # Pop from heap until we find a valid (non-stale) entry
            best_pair = None
            while heap:
                neg_cnt, pair = heapq.heappop(heap)
                if pair in pair_counts and pair_counts[pair] == -neg_cnt:
                    best_pair = pair
                    best_count = -neg_cnt
                    break

            if best_pair is None:
                if verbose:
                    print(f"No more pairs at iteration {merge_idx}")
                break

            if best_count < self.min_frequency:
                if verbose:
                    print(f"Best pair freq {best_count} < min_freq {self.min_frequency}, stopping")
                break

            a, b = best_pair
            merged_seg = id_to_seg[a] + id_to_seg[b]
            merged_id = get_id(merged_seg)

            # Record
            self.merges.append(((id_to_seg[a], id_to_seg[b]), f"<SEG_{merge_idx}>"))
            self.segments.append(BPESegment(
                id=merge_idx, tokens=merged_seg, frequency=best_count
            ))

            # Only visit sequences that contain this pair
            del pair_counts[(a, b)]
            affected = pair_to_seqs.pop((a, b), set())

            for seq_idx in affected:
                seq = int_seqs[seq_idx]
                i = 0
                while i < len(seq) - 1:
                    if seq[i] != a or seq[i + 1] != b:
                        i += 1
                        continue

                    # Left neighbor
                    if i > 0:
                        left = seq[i - 1]
                        old_p = (left, a)
                        pair_counts[old_p] = pair_counts.get(old_p, 0) - 1
                        if pair_counts[old_p] <= 0:
                            pair_counts.pop(old_p, None)
                        new_p = (left, merged_id)
                        new_cnt = pair_counts.get(new_p, 0) + 1
                        pair_counts[new_p] = new_cnt
                        if new_p not in pair_to_seqs:
                            pair_to_seqs[new_p] = set()
                        pair_to_seqs[new_p].add(seq_idx)
                        heapq.heappush(heap, (-new_cnt, new_p))

                    # Right neighbor
                    if i + 2 < len(seq):
                        right = seq[i + 2]
                        old_p = (b, right)
                        pair_counts[old_p] = pair_counts.get(old_p, 0) - 1
                        if pair_counts[old_p] <= 0:
                            pair_counts.pop(old_p, None)
                        new_p = (merged_id, right)
                        new_cnt = pair_counts.get(new_p, 0) + 1
                        pair_counts[new_p] = new_cnt
                        if new_p not in pair_to_seqs:
                            pair_to_seqs[new_p] = set()
                        pair_to_seqs[new_p].add(seq_idx)
                        heapq.heappush(heap, (-new_cnt, new_p))

                    seq[i] = merged_id
                    del seq[i + 1]

            if verbose and (merge_idx + 1) % 50 == 0:
                tokens_str = ' '.join(merged_seg[:6])
                if len(merged_seg) > 6:
                    tokens_str += f"... ({len(merged_seg)} tokens)"
                print(f"  Merge {merge_idx + 1}: freq={best_count}, {tokens_str}")

        if verbose:
            print(f"Training complete. Learned {len(self.segments)} segments.")

        self._encode_index = None  # invalidate cache
        return self.segments

    # =========================================================================
    # Encode / Decode
    # =========================================================================

    def encode(self, tokens: List[str]) -> List[str]:
        """
        Encode token sequence using learned BPE.

        Uses first-token index for O(1) candidate lookup instead of scanning all segments.
        """
        index = self._build_encode_index()
        result = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Preserve structure tokens
            if token in self.STRUCTURE_TOKENS:
                result.append(token)
                i += 1
                continue

            # Preserve attribute tokens
            if '=' in token and not token.startswith('<'):
                result.append(token)
                i += 1
                continue

            # Lookup candidates by first token
            candidates = index.get(token)
            if candidates:
                matched = False
                for seg_tokens, seg_id in candidates:
                    seg_len = len(seg_tokens)
                    if i + seg_len <= len(tokens) and tuple(tokens[i:i + seg_len]) == seg_tokens:
                        result.append(f"<SEG_{seg_id}>")
                        i += seg_len
                        matched = True
                        break
                if not matched:
                    result.append(token)
                    i += 1
            else:
                result.append(token)
                i += 1

        return result

    def _find_segment_id(self, segment: Tuple[str, ...]) -> Optional[int]:
        """Find segment ID for a token tuple, or None if not learned."""
        for seg in self.segments:
            if seg.tokens == segment:
                return seg.id
        return None

    def decode(self, tokens: List[str]) -> List[str]:
        """
        Decode merged tokens back to atomic tokens.

        Args:
            tokens: Level 2 tokens with SEG_N tokens

        Returns:
            Level 1 atomic tokens
        """
        result = []
        for token in tokens:
            if token.startswith('<SEG_') and token.endswith('>'):
                seg_id = int(token[5:-1])
                for seg in self.segments:
                    if seg.id == seg_id:
                        result.extend(seg.tokens)
                        break
            else:
                result.append(token)
        return result

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, path: str):
        """Save BPE model to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "coord_range": self.coord_range,
            "relative_range": self.relative_range,
            "min_frequency": self.min_frequency,
            "merges": [
                ([list(p) for p in pair], seg_token)
                for pair, seg_token in self.merges
            ],
            "segments": [
                {
                    "id": seg.id,
                    "tokens": list(seg.tokens),
                    "frequency": seg.frequency,
                }
                for seg in self.segments
            ]
        }
        with output_path.open('w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SVGBPETrainer':
        """Load BPE model from JSON."""
        with open(path) as f:
            data = json.load(f)

        trainer = cls(
            coord_range=data.get("coord_range", 234),
            relative_range=data.get("relative_range", data.get("coord_range", 234)),
            min_frequency=data.get("min_frequency", 5),
        )

        trainer.merges = [
            ((tuple(pair[0]), tuple(pair[1])), seg_token)
            for pair, seg_token in data["merges"]
        ]

        trainer.segments = [
            BPESegment(
                id=seg["id"],
                tokens=tuple(seg["tokens"]),
                frequency=seg["frequency"],
            )
            for seg in data["segments"]
        ]

        return trainer

    def get_segment_tokens(self) -> List[str]:
        """Get list of all learned segment token names."""
        return [seg.token_name for seg in self.segments]

    def print_top_segments(self, n: int = 20):
        """Print top N segments by frequency."""
        print(f"\nTop {n} Learned Segments (complete commands only):")
        print("=" * 80)
        print(f"{'ID':<6} {'Token':<12} {'Freq':<8} {'Original Sequence'}")
        print("-" * 80)

        sorted_segments = sorted(self.segments, key=lambda s: s.frequency, reverse=True)
        for seg in sorted_segments[:n]:
            tokens_str = ' '.join(seg.tokens)
            if len(tokens_str) > 48:
                tokens_str = tokens_str[:45] + "..."
            print(f"{seg.id:<6} {seg.token_name:<12} {seg.frequency:<8} {tokens_str}")
