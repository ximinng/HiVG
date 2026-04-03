"""Utilities for converting atomic-token SVG payloads to segment-token JSONL exports."""

import errno
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from .bpe import SVGBPETrainer
from .tokenizer import string_to_tokens, tokens_to_string


def encode_segment_tokens(bpe: SVGBPETrainer, atomic_token_str: str) -> str:
    """Atomic-token smart-space string -> segment-token smart-space string."""
    atomic_tokens = string_to_tokens(atomic_token_str)
    segment_tokens = bpe.encode(atomic_tokens)
    return tokens_to_string(segment_tokens)


def contains_segment_token(segment_token_str: str) -> bool:
    """Return True when encoded output contains at least one learned segment token."""
    return "<SEG_" in segment_token_str


_SIZE_SPEC_PATTERN = re.compile(r"^\s*(\d+)\s*([wWmMkK]?)\s*$")


def parse_size_spec(size_spec: str) -> int:
    """Parse size strings like 50000, 5w, 2m into an integer count."""
    if not size_spec:
        raise ValueError("size_spec must be non-empty")

    match = _SIZE_SPEC_PATTERN.match(size_spec)
    if not match:
        raise ValueError(
            f"Invalid size spec `{size_spec}`. Use plain int or suffix one of k/K/w/W/m/M."
        )

    value = int(match.group(1))
    suffix = match.group(2).lower()
    multipliers = {
        "": 1,
        "k": 1_000,
        "w": 10_000,
        "m": 1_000_000,
    }
    return value * multipliers[suffix]


def write_line_with_retry(
    file_obj,
    file_path: Path,
    line: str,
    retries: int = 5,
    retry_wait: float = 1.0,
):
    """Write one line with ESTALE retry by reopening file in append mode."""
    for attempt in range(retries + 1):
        try:
            file_obj.write(line)
            return file_obj
        except OSError as e:
            if e.errno != errno.ESTALE or attempt >= retries:
                raise

            try:
                file_obj.close()
            except Exception:
                pass

            wait_s = retry_wait * (attempt + 1)
            print(
                f"[Warning] Stale file handle on {file_path}, "
                f"reopen and retry ({attempt + 1}/{retries}) after {wait_s:.1f}s..."
            )
            time.sleep(wait_s)
            file_obj = open(file_path, "a", encoding="utf-8")

    return file_obj


def encode_text2svg_record(
    record: Dict[str, Any], bpe: SVGBPETrainer
) -> Tuple[Dict[str, Any], int, int]:
    """Convert one Alpaca-style text2svg record from atomic tokens to segment tokens."""
    atomic_token_str = record.get("output")
    if not isinstance(atomic_token_str, str) or not atomic_token_str:
        raise ValueError("text2svg JSONL record is missing a non-empty `output` field.")

    segment_token_str = encode_segment_tokens(bpe, atomic_token_str)
    updated = dict(record)
    updated["output"] = segment_token_str
    return (
        updated,
        len(string_to_tokens(atomic_token_str)),
        len(string_to_tokens(segment_token_str)),
    )


def _find_last_assistant_index(messages: List[Dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return idx
    return -1


def _extract_svg_content(record: Dict[str, Any]) -> str:
    """Extract the SVG token payload from supported JSONL record shapes."""
    output = record.get("output")
    if isinstance(output, str):
        return output

    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""

    assistant_idx = _find_last_assistant_index(messages)
    if assistant_idx < 0:
        return ""

    assistant_msg = messages[assistant_idx]
    if not isinstance(assistant_msg, dict):
        return ""

    content = assistant_msg.get("content")
    return content if isinstance(content, str) else ""


def encode_image2svg_record(
    record: Dict[str, Any], bpe: SVGBPETrainer
) -> Tuple[Dict[str, Any], int, int]:
    """Convert one ShareGPT-style image2svg record from atomic tokens to segment tokens."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("image2svg JSONL record is missing a `messages` list.")

    assistant_idx = _find_last_assistant_index(messages)
    if assistant_idx < 0:
        raise ValueError("image2svg JSONL record does not contain an assistant message.")

    assistant_msg = messages[assistant_idx]
    atomic_token_str = assistant_msg.get("content") if isinstance(assistant_msg, dict) else None
    if not isinstance(atomic_token_str, str) or not atomic_token_str:
        raise ValueError("image2svg assistant message is missing non-empty `content`.")

    segment_token_str = encode_segment_tokens(bpe, atomic_token_str)
    updated_messages = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
    updated_messages[assistant_idx]["content"] = segment_token_str

    updated = dict(record)
    updated["messages"] = updated_messages
    return (
        updated,
        len(string_to_tokens(atomic_token_str)),
        len(string_to_tokens(segment_token_str)),
    )


def resolve_jsonl_output_path(
    task_name: str,
    input_path: str,
    explicit_output: str = "",
    output_dir: str = "",
    output_prefix: str = "",
) -> Path:
    """Resolve output path for direct JSONL conversion."""
    if explicit_output:
        return Path(explicit_output)

    input_file = Path(input_path)
    base_dir = Path(output_dir) if output_dir else input_file.parent
    if output_prefix:
        return base_dir / f"{output_prefix}_{task_name}.jsonl"
    return base_dir / f"{input_file.stem}_segment.jsonl"


def convert_jsonl_file(
    input_path: str,
    output_path: str,
    transform_record: Callable[[Dict[str, Any], SVGBPETrainer], Tuple[Dict[str, Any], int, int]],
    bpe: SVGBPETrainer,
    task_name: str,
    retries: int = 5,
    retry_wait: float = 1.0,
    flush_every: int = 2000,
    target_records: Optional[int] = None,
    require_segment_match: bool = False,
) -> Dict[str, Any]:
    """Stream-convert an atomic-token JSONL file into a segment-token JSONL file."""
    src = Path(input_path)
    dst = Path(output_path)
    if src.resolve() == dst.resolve():
        raise ValueError(f"{task_name}: input and output paths must differ: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    total_atomic_tokens = 0
    total_segment_tokens = 0
    blank_lines = 0
    skipped_no_segment_match = 0
    seen_records = 0

    if target_records is not None and target_records <= 0:
        with open(dst, "w", encoding="utf-8"):
            pass
        return {
            "task": task_name,
            "input_path": str(src),
            "output_path": str(dst),
            "seen_records": seen_records,
            "written": written,
            "blank_lines": blank_lines,
            "skipped_no_segment_match": skipped_no_segment_match,
            "require_segment_match": require_segment_match,
            "target_records": target_records,
            "atomic_tokens": total_atomic_tokens,
            "segment_tokens": total_segment_tokens,
            "compression": 0.0,
        }

    with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for line_no, raw_line in enumerate(tqdm(f_in, desc=f"Encoding {task_name}"), 1):
            if not raw_line.strip():
                blank_lines += 1
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{task_name}: failed to parse JSON at line {line_no} in {src}: {exc}"
                ) from exc

            converted, atomic_token_count, segment_token_count = transform_record(record, bpe)
            seen_records += 1
            has_segment_match = contains_segment_token(_extract_svg_content(converted))

            if require_segment_match and not has_segment_match:
                skipped_no_segment_match += 1
                continue

            out_line = json.dumps(converted, ensure_ascii=False) + "\n"
            f_out = write_line_with_retry(
                f_out,
                dst,
                out_line,
                retries=retries,
                retry_wait=retry_wait,
            )
            written += 1
            total_atomic_tokens += atomic_token_count
            total_segment_tokens += segment_token_count

            if flush_every > 0 and written % flush_every == 0:
                f_out.flush()

            if target_records is not None and written >= target_records:
                break

    compression = (
        (1 - total_segment_tokens / total_atomic_tokens) * 100 if total_atomic_tokens else 0.0
    )
    return {
        "task": task_name,
        "input_path": str(src),
        "output_path": str(dst),
        "seen_records": seen_records,
        "written": written,
        "blank_lines": blank_lines,
        "skipped_no_segment_match": skipped_no_segment_match,
        "require_segment_match": require_segment_match,
        "target_records": target_records,
        "atomic_tokens": total_atomic_tokens,
        "segment_tokens": total_segment_tokens,
        "compression": compression,
    }
