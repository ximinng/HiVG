"""Shared dataset-size guardrails for BPE training scripts."""

from dataclasses import dataclass
from typing import Optional

# Keep the default below the largest published ablation size in this repo.
AUTO_SAMPLE_THRESHOLD = 1_500_000
DEFAULT_MAX_TRAINING_SAMPLES = 1_200_000


@dataclass(frozen=True)
class SamplePlan:
    """How a script should bound the dataset before BPE training."""

    effective_max_samples: Optional[int]
    warning: Optional[str] = None


def choose_bpe_sample_plan(
    total_samples: int,
    requested_max_samples: Optional[int],
    force_large_run: bool,
    auto_sample_threshold: int = AUTO_SAMPLE_THRESHOLD,
    default_max_training_samples: int = DEFAULT_MAX_TRAINING_SAMPLES,
) -> SamplePlan:
    """Choose a safe default sample cap for exact in-memory BPE training."""
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")

    if requested_max_samples is not None and requested_max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")

    effective_requested = requested_max_samples
    if effective_requested is not None:
        effective_requested = min(effective_requested, total_samples)

    if force_large_run:
        risky_samples = effective_requested if effective_requested is not None else total_samples
        if risky_samples > auto_sample_threshold:
            return SamplePlan(
                effective_max_samples=effective_requested,
                warning=(
                    "Warning: forcing exact BPE training on a large dataset "
                    f"({risky_samples} samples). This path still keeps full "
                    "sequence and pair state in memory and may be killed by OOM."
                ),
            )
        return SamplePlan(effective_max_samples=effective_requested)

    if effective_requested is not None:
        if effective_requested > auto_sample_threshold:
            raise ValueError(
                "--max-samples exceeds the safe default threshold for exact in-memory "
                f"BPE training ({effective_requested} > {auto_sample_threshold}). "
                "Lower --max-samples or pass --force-large-run if you really want "
                "to attempt it."
            )
        return SamplePlan(effective_max_samples=effective_requested)

    if total_samples <= auto_sample_threshold:
        return SamplePlan(effective_max_samples=None)

    auto_cap = min(default_max_training_samples, total_samples)
    return SamplePlan(
        effective_max_samples=auto_cap,
        warning=(
            "Warning: dataset is large for exact in-memory BPE training "
            f"({total_samples} samples). Automatically limiting training to the first "
            f"{auto_cap} samples. Full-dataset exact training is not required to learn "
            "the dominant SVG command patterns."
        ),
    )
