from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from tetrnn.target_audit import InputSpec, audit_directory, main, parse_input_spec


def _write_shard(directory: Path, *, all_net_seats: bool = False) -> None:
    directory.mkdir()
    decision = np.zeros((4, 8), dtype=np.int32)
    # Net seats are seat == game_id % 2: decisions 0 and 2.
    decision[:, :2] = np.array(
        [[0, 0], [1, 1], [2, 0], [3, 1]] if all_net_seats else [[0, 0], [0, 1], [1, 1], [1, 0]],
        dtype=np.int32,
    )
    scores = np.array(
        [
            0,
            10,
            5,
            5,
            -1_000_000,
            -2_000_000,
            -2_000_001,
            -2_000_002,
            7,
            7,
        ],
        dtype=np.int32,
    )
    children = scores.size
    save_file(
        {
            "decision": decision,
            "opp_plane": np.zeros((4, 50), dtype=np.uint8),
            "child_offset": np.array([0, 2, 5, 8, 10], dtype=np.int32),
            "child_own": np.zeros((children, 50), dtype=np.uint8),
            "child_feats": np.zeros((children, 85), dtype=np.float32),
            "child_score": scores,
        },
        directory / "shard-00000.safetensors",
    )


def test_parse_input_spec_uses_only_a_known_trailing_selector() -> None:
    assert parse_input_spec("r0=/tmp/corpus") == InputSpec("r0", "/tmp/corpus", "all")
    assert parse_input_spec("net=/tmp/corpus:net-seat") == InputSpec(
        "net", "/tmp/corpus", "net-seat"
    )
    assert parse_input_spec("literal=/tmp/a:unknown") == InputSpec(
        "literal", "/tmp/a:unknown", "all"
    )
    with pytest.raises(argparse.ArgumentTypeError, match="LABEL=DIR"):
        parse_input_spec("missing-equals")


def test_audit_metrics_and_seat_filters(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_shard(corpus)

    report = audit_directory(InputSpec("all", str(corpus), "all"))
    assert report["decisions"] == 4
    assert report["children"] == {
        "total": 10,
        "per_decision_quantiles": {
            "p0": 2.0,
            "p10": 2.0,
            "p25": 2.0,
            "p50": 2.5,
            "p75": 3.0,
            "p90": 3.0,
            "p95": 3.0,
            "p99": 3.0,
            "p100": 3.0,
        },
    }
    assert report["ties"] == {
        "top_tie_decisions": 2,
        "top_tie_fraction": 0.5,
        "all_tied_decisions": 1,
        "all_tied_fraction": 0.25,
    }
    assert report["terminal_sentinel"] == {
        "score_lte": -1_000_000.0,
        "children": 4,
        "any_decisions": 2,
        "any_fraction": 0.5,
        "all_decisions": 1,
        "all_fraction": 0.25,
    }
    assert report["unique_top_gap"] == {
        "decisions": 2,
        "absolute_median": 5.5,
        "normalized_median": 0.75,
    }
    target = report["target"]
    assert target["targetable_decisions"] == 3
    assert target["rejected_partial_terminal"] == 1
    assert target["fraction_top1_ge_0_9"] == pytest.approx(1 / 3)
    assert target["fraction_n_eff_lt_2_5"] == pytest.approx(2 / 3)

    net = audit_directory(InputSpec("net", str(corpus), "net-seat"))
    teacher = audit_directory(InputSpec("teacher", str(corpus), "teacher-seat"))
    assert net["decisions"] == teacher["decisions"] == 2
    assert net["terminal_sentinel"]["all_decisions"] == 1
    assert teacher["terminal_sentinel"]["all_decisions"] == 0
    assert net["ties"]["top_tie_decisions"] == 0
    assert teacher["ties"]["top_tie_decisions"] == 2
    assert net["target"]["rejected_partial_terminal"] == 0
    assert teacher["target"]["rejected_partial_terminal"] == 1


def test_cli_json_is_byte_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus"
    _write_shard(corpus)
    argument = f"fixture={corpus}:net-seat"

    assert main([argument]) == 0
    first = capsys.readouterr().out
    assert main([argument]) == 0
    second = capsys.readouterr().out

    assert first == second
    parsed = json.loads(first)
    assert parsed["target_contract"]["generator_logits"] == "uniform-zero"
    assert parsed["target_contract"]["legacy_forensic_only"] is True
    assert parsed["corpora"]["fixture"]["selector"] == "net-seat"


def test_empty_selection_is_valid_json_without_nan(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_shard(corpus, all_net_seats=True)
    report = audit_directory(InputSpec("teacher", str(corpus), "teacher-seat"))
    assert report["decisions"] == 0
    assert report["children"]["per_decision_quantiles"] is None
    assert report["target"]["fraction_top1_ge_0_9"] is None
    assert json.dumps(report, allow_nan=False)


def test_missing_shards_fails_with_context(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no shard-\*\.safetensors"):
        audit_directory(InputSpec("empty", str(tmp_path), "all"))


def test_parallel_worker_subdirectories_are_audited_recursively(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    _write_shard(root / "w0")
    _write_shard(root / "w1")
    report = audit_directory(InputSpec("workers", str(root), "all"))
    assert report["shards"] == 2
    assert report["decisions"] == 8
