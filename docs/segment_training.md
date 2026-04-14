# Segment Token Training Guide

This guide walks through training **BPE segment tokens** from an SVG corpus using the HiVG tokenizer.
Segment tokens are the key to HiVG's 2.76x token compression — they merge frequently co-occurring SVG command sequences into single tokens.

## Background

HiVG uses a three-level tokenization hierarchy:

| Level | Name | Example |
|:---:|:---|:---|
| 0 | Raw SVG | `M 100 200 c -5 0 -10 5 -10 10` |
| 1 | Atomic tokens | `<cmd_M><P_100><P_200><cmd_c><d_-5><d_0><d_-10><d_5><d_-10><d_10>` |
| 2 | Segment tokens | `<cmd_M><P_100><P_200><SEG_42>` |

Level 2 is learned by running **semantic BPE** on Level 1 sequences.
Unlike standard BPE, this variant only merges **complete path command segments** (e.g., a full cubic Bézier `<cmd_c>` + 6 coordinates), so every learned merge is a valid SVG operation.

## Prerequisites

```bash
pip install -e .
```

All required dependencies (`pyyaml`, `tqdm`, etc.) are included in `setup.py`.

## Step 1 — Prepare Atomic Token Data

Start with a JSONL file where each line contains SVG content encoded as **Level 1 atomic tokens**.
Two formats are supported:

**Alpaca format** (text2svg):

```jsonl
{"instruction": "Draw a red circle", "input": "", "output": "<svg><circle>...</circle></svg>"}
```

**ShareGPT format** (img2svg):

```jsonl
{"messages": [{"role": "user", "content": "<image>Convert to SVG"}, {"role": "assistant", "content": "<svg><path>...</path></svg>"}], "images": ["img.png"]}
```

The `output` field (Alpaca) or last `assistant` message (ShareGPT) should contain the atomic token string.

### Tokenizing raw SVG

If you have raw SVG files, tokenize them first:

```python
from hivg_tokenizer import SVGTokenizer, tokens_to_string

tokenizer = SVGTokenizer(coord_range=234)

with open("example.svg") as f:
    svg_string = f.read()

tokens = tokenizer.tokenize(svg_string)
token_str = tokens_to_string(tokens)  # compact string representation
print(token_str)
# '<svg><path><cmd_M><P_100><P_200><cmd_c><d_-5><d_0>...'
```

## Step 2 — Train BPE Segments

### 2.1 Load training data

```python
import json
from hivg_tokenizer import SVGBPETrainer, string_to_tokens

# Collect atomic token sequences from your JSONL corpus
token_sequences = []
with open("train_data.jsonl") as f:
    for line in f:
        record = json.loads(line)
        atomic_str = record.get("output", "")  # Alpaca format
        if atomic_str:
            token_sequences.append(string_to_tokens(atomic_str))

print(f"Loaded {len(token_sequences)} sequences")
```

### 2.2 Train

```python
trainer = SVGBPETrainer(
    coord_range=234,       # must match your tokenizer config
    min_frequency=50,      # minimum pair frequency to learn a merge
)

segments = trainer.train(
    token_sequences,
    num_merges=500,        # number of BPE merge operations
    verbose=True,
)
```

Training output:

```
Extracting complete segments...
Extracted 4821630 complete segments from 120000 sequences
Starting BPE training (max 500 merges, min_freq=50)...
Unique segments: 28451
Unique pairs: 193720
  Merge 50: freq=12340, <cmd_l> <d_0> <d_-1>... (3 tokens)
  Merge 100: freq=8210, <cmd_c> <d_0> <d_3> <d_-2> <d_5> <d_-2> <d_3>
  ...
Training complete. Learned 500 segments.
```

### 2.3 Inspect results

```python
trainer.print_top_segments(n=20)
```

```
Top 20 Learned Segments (complete commands only):
================================================================================
ID     Token        Freq     Original Sequence
--------------------------------------------------------------------------------
42     <SEG_42>     18320    <cmd_c> <d_0> <d_3> <d_-2> <d_5> <d_-2> <d_3>
17     <SEG_17>     15890    <cmd_l> <d_0> <d_-1>
...
```

### 2.4 Save the model

```python
trainer.save("bpe_model.json")
```

The JSON file contains all merge rules and segment definitions, and can be loaded later:

```python
trainer = SVGBPETrainer.load("bpe_model.json")
```

## Step 3 — Encode & Decode

### Single sequence

```python
from hivg_tokenizer import string_to_tokens, tokens_to_string

trainer = SVGBPETrainer.load("bpe_model.json")

atomic_tokens = string_to_tokens("<svg><path><cmd_M><P_50><P_50><cmd_c><d_0><d_3>...")
segment_tokens = trainer.encode(atomic_tokens)  # Level 1 → Level 2

print(f"Atomic:  {len(atomic_tokens)} tokens")
print(f"Segment: {len(segment_tokens)} tokens")
print(f"Compression: {len(atomic_tokens) / len(segment_tokens):.2f}x")

# Decode back to atomic tokens
recovered = trainer.decode(segment_tokens)
assert recovered == atomic_tokens
```

### Batch JSONL conversion

The `segment_io` module provides utilities for converting entire JSONL datasets:

```python
from hivg_tokenizer.bpe import SVGBPETrainer
from hivg_tokenizer.segment_io import (
    convert_jsonl_file,
    encode_text2svg_record,
    encode_image2svg_record,
)

trainer = SVGBPETrainer.load("bpe_model.json")

# Text-to-SVG dataset (Alpaca format)
stats = convert_jsonl_file(
    input_path="train_text2svg.jsonl",
    output_path="train_text2svg_segment.jsonl",
    transform_record=encode_text2svg_record,
    bpe=trainer,
    task_name="text2svg",
)
print(f"Written: {stats['written']} records")
print(f"Compression: {stats['compression']:.1f}%")

# Image-to-SVG dataset (ShareGPT format)
stats = convert_jsonl_file(
    input_path="train_img2svg.jsonl",
    output_path="train_img2svg_segment.jsonl",
    transform_record=encode_image2svg_record,
    bpe=trainer,
    task_name="img2svg",
)
```

#### Conversion options

| Parameter | Default | Description |
|:---|:---:|:---|
| `target_records` | `None` | Stop after N records (useful for quick tests) |
| `require_segment_match` | `False` | Skip records where no segment token was applied |
| `flush_every` | `2000` | Flush output file every N records |
| `retries` | `5` | Retry count on stale file handle (NFS/FUSE) |

## Configuration Reference

Pre-defined configs are located in `hivg_tokenizer/configs/`:

| Config | Canvas | `coord_range` | `relative_range` | BPE merges | Min freq |
|:---|:---:|:---:|:---:|:---:|:---:|
| `config_224.yaml` | 224 | 234 | 234 | 500 | 50 |
| `config_784.yaml` | 784 | 794 | 794 | 1000 | 100 |
| `config_1008.yaml` | 1024 | 1008 | 1008 | 500 | 10 |

Loading a config:

```python
import yaml

with open("hivg_tokenizer/configs/config_224.yaml") as f:
    cfg = yaml.safe_load(f)

trainer = SVGBPETrainer(
    coord_range=cfg["coord_range"],
    relative_range=cfg["relative_range"],
    min_frequency=cfg["bpe"]["min_frequency"],
)
```

## Large Dataset Training

For datasets exceeding ~1.5M samples, the built-in guardrail in `training_limits.py` automatically caps the training set to 1.2M samples.
This is sufficient to learn dominant SVG patterns without exhausting memory.

To override:

```python
from hivg_tokenizer.training_limits import choose_bpe_sample_plan

plan = choose_bpe_sample_plan(
    total_samples=3_000_000,
    requested_max_samples=None,
    force_large_run=True,         # bypass the auto-cap
)
```

## End-to-End Example

```python
import json
from hivg_tokenizer import SVGTokenizer, SVGBPETrainer, string_to_tokens, tokens_to_string

# 1. Tokenize raw SVGs to atomic tokens
tokenizer = SVGTokenizer(coord_range=234)
token_sequences = []

for svg_file in svg_files:
    with open(svg_file) as f:
        tokens = tokenizer.tokenize(f.read())
        token_sequences.append(tokens)

# 2. Train BPE
trainer = SVGBPETrainer(coord_range=234, min_frequency=50)
trainer.train(token_sequences, num_merges=500)
trainer.save("bpe_model.json")

# 3. Encode a sequence
atomic = token_sequences[0]
segment = trainer.encode(atomic)
print(f"{len(atomic)} → {len(segment)} tokens ({len(atomic)/len(segment):.2f}x)")

# 4. Verify round-trip
assert trainer.decode(segment) == atomic
```

## FAQ

**Q: How many training samples do I need?**
A: 50k–200k SVG samples are typically enough to learn stable segment patterns. More data improves coverage of rare command combinations.

**Q: How do I choose `num_merges`?**
A: 500 merges is a good default. More merges give higher compression but add more special tokens to the LLM vocabulary. The sweet spot depends on your corpus diversity.

**Q: Should `coord_range` match at training and inference time?**
A: Yes. The BPE model is trained with a specific `coord_range`, and the same value must be used during encoding, decoding, and inference. A mismatch will produce incorrect SVG output.

**Q: Can I merge segment models trained on different datasets?**
A: No. Each BPE model's segment IDs are assigned sequentially during training. Use a single training run on a combined corpus instead.
