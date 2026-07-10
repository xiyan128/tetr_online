"""Audit legacy beam-rank targets without modifying their shard corpus.

Legacy shards do not contain frozen generator logits, so this forensic audit
uses an explicitly uniform generator (zero logits) with the current
``rank_distillation_target`` transform.  Its output is deterministic JSON and
is intended to make target-collapse and score-shape evidence reproducible; it
does not make a legacy shard promotion-eligible.

Usage::

    python -m tetrnn.target_audit \
        round0=/path/to/corpus \
        r11-net=/path/to/r11/corpus:net-seat

The optional selector is one of ``all``, ``net-seat``, or ``teacher-seat``.
For two-arm legacy games the net seat is ``game_id % 2`` and the teacher seat
is the opposite seat.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .shards import read_shard
from .targets import (
    BEAM_RANK_BETA,
    LEGACY_TERMINAL_THRESHOLD,
    n_eff,
    rank_distillation_target,
)

SeatSelector = Literal["all", "net-seat", "teacher-seat"]
SEAT_SELECTORS: tuple[SeatSelector, ...] = ("all", "net-seat", "teacher-seat")
TERMINAL_SENTINEL = LEGACY_TERMINAL_THRESHOLD
QUANTILES: tuple[tuple[str, float], ...] = (
    ("p0", 0.0),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.5),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p100", 1.0),
)


@dataclass(frozen=True)
class InputSpec:
    label: str
    directory: str
    selector: SeatSelector


@dataclass
class _Accumulator:
    decisions: int = 0
    targetable_decisions: int = 0
    rejected_partial_terminal: int = 0
    children: int = 0
    child_counts: list[int] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    effective_supports: list[float] = field(default_factory=list)
    top1: list[float] = field(default_factory=list)
    top1_ge_09: int = 0
    effective_support_lt_25: int = 0
    top_tie: int = 0
    all_tied: int = 0
    sentinel_children: int = 0
    sentinel_any: int = 0
    sentinel_all: int = 0
    unique_top_absolute_gaps: list[float] = field(default_factory=list)
    unique_top_normalized_gaps: list[float] = field(default_factory=list)


def parse_input_spec(value: str) -> InputSpec:
    """Parse ``LABEL=DIR[:SELECTOR]`` without confusing ordinary colons."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must have the form LABEL=DIR[:SELECTOR]")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("input label must not be empty")
    if not raw_path:
        raise argparse.ArgumentTypeError(f"directory for {label!r} must not be empty")

    directory = raw_path
    selector: SeatSelector = "all"
    maybe_path, separator, maybe_selector = raw_path.rpartition(":")
    if separator and maybe_selector in SEAT_SELECTORS:
        directory = maybe_path
        selector = cast(SeatSelector, maybe_selector)
    if not directory:
        raise argparse.ArgumentTypeError(f"directory for {label!r} must not be empty")
    return InputSpec(label=label, directory=directory, selector=selector)


def _selected(game_id: int, seat: int, selector: SeatSelector) -> bool:
    if selector == "all":
        return True
    is_net_seat = seat == game_id % 2
    return is_net_seat if selector == "net-seat" else not is_net_seat


def _quantiles(values: list[int] | list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, probability, method="linear"))
        for name, probability in QUANTILES
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _record(scores: np.ndarray, accumulator: _Accumulator) -> None:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("every audited decision must have at least one child score")

    accumulator.decisions += 1
    accumulator.children += int(scores.size)
    accumulator.child_counts.append(int(scores.size))

    top_count = int(np.count_nonzero(scores == scores.max()))
    accumulator.top_tie += int(top_count > 1)
    accumulator.all_tied += int(np.all(scores == scores[0]))

    sentinel = scores <= TERMINAL_SENTINEL
    accumulator.sentinel_children += int(sentinel.sum())
    accumulator.sentinel_any += int(sentinel.any())
    accumulator.sentinel_all += int(sentinel.all())

    partial_terminal = bool(sentinel.any() and not sentinel.all())
    if partial_terminal:
        accumulator.rejected_partial_terminal += 1
    else:
        target = rank_distillation_target(scores, np.zeros_like(scores))
        positive = target[target > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        support = n_eff(target)
        top_probability = float(target.max())
        accumulator.targetable_decisions += 1
        accumulator.entropies.append(entropy)
        accumulator.effective_supports.append(support)
        accumulator.top1.append(top_probability)
        accumulator.top1_ge_09 += int(top_probability >= 0.9)
        accumulator.effective_support_lt_25 += int(support < 2.5)

    # Gap statistics are meaningful only with a unique best and a runner-up.
    if scores.size >= 2 and top_count == 1:
        ordered = np.sort(scores)
        gap = float(ordered[-1] - ordered[-2])
        span = float(ordered[-1] - ordered[0])
        accumulator.unique_top_absolute_gaps.append(gap)
        accumulator.unique_top_normalized_gaps.append(gap / span)


def audit_directory(spec: InputSpec) -> dict[str, Any]:
    """Stream one directory's shards and return JSON-serializable metrics."""
    # Parallel datagen writes one worker subdirectory per process; audit the
    # dataset root recursively so a report cannot silently cover only w0.
    paths = sorted(str(path) for path in Path(spec.directory).rglob("shard-*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no shard-*.safetensors files in {spec.directory}")

    accumulator = _Accumulator()
    for path in paths:
        shard = read_shard(path)
        for decision_index, row in enumerate(shard.decision):
            game_id, seat = int(row[0]), int(row[1])
            if _selected(game_id, seat, spec.selector):
                _record(shard.child_score[shard.children_of(decision_index)], accumulator)

    n = accumulator.decisions
    targetable = accumulator.targetable_decisions
    unique_top_count = len(accumulator.unique_top_absolute_gaps)
    return {
        "directory": str(Path(spec.directory)),
        "selector": spec.selector,
        "shards": len(paths),
        "decisions": n,
        "children": {
            "total": accumulator.children,
            "per_decision_quantiles": _quantiles(accumulator.child_counts),
        },
        "target": {
            "targetable_decisions": targetable,
            "rejected_partial_terminal": accumulator.rejected_partial_terminal,
            "partial_terminal_score_lte": TERMINAL_SENTINEL,
            "entropy_quantiles": _quantiles(accumulator.entropies),
            "n_eff_quantiles": _quantiles(accumulator.effective_supports),
            "top1_quantiles": _quantiles(accumulator.top1),
            "fraction_top1_ge_0_9": _fraction(accumulator.top1_ge_09, targetable),
            "fraction_n_eff_lt_2_5": _fraction(accumulator.effective_support_lt_25, targetable),
        },
        "ties": {
            "top_tie_decisions": accumulator.top_tie,
            "top_tie_fraction": _fraction(accumulator.top_tie, n),
            "all_tied_decisions": accumulator.all_tied,
            "all_tied_fraction": _fraction(accumulator.all_tied, n),
        },
        "terminal_sentinel": {
            "score_lte": TERMINAL_SENTINEL,
            "children": accumulator.sentinel_children,
            "any_decisions": accumulator.sentinel_any,
            "any_fraction": _fraction(accumulator.sentinel_any, n),
            "all_decisions": accumulator.sentinel_all,
            "all_fraction": _fraction(accumulator.sentinel_all, n),
        },
        "unique_top_gap": {
            "decisions": unique_top_count,
            "absolute_median": _median(accumulator.unique_top_absolute_gaps),
            "normalized_median": _median(accumulator.unique_top_normalized_gaps),
        },
    }


def build_report(specs: list[InputSpec]) -> dict[str, Any]:
    labels = [spec.label for spec in specs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate input labels: {', '.join(duplicates)}")
    return {
        "schema_version": 1,
        "target_contract": {
            "function": "rank_distillation_target",
            "generator_logits": "uniform-zero",
            "beta": BEAM_RANK_BETA,
            "legacy_forensic_only": True,
        },
        "corpora": {spec.label: audit_directory(spec) for spec in specs},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        metavar="LABEL=DIR[:SELECTOR]",
        type=parse_input_spec,
        nargs="+",
        help="legacy shard directory and optional all/net-seat/teacher-seat filter",
    )
    args = parser.parse_args(argv)
    try:
        report = build_report(args.inputs)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
