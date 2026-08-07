import json
import pytest

from benchmark import run_benchmark


def test_benchmark_produces_well_formed_results(tmp_path):
    summary = run_benchmark(n=3, tolerance_px=5.0, out_dir=tmp_path,
                             n_examples=1, seed_start=1000)

    assert summary["n_samples"] == 3
    assert 0 <= summary["n_within_tolerance"] <= 3
    assert 0 <= summary["n_success"] <= 3
    assert "success_rate_pct" in summary

    results_path = tmp_path / "benchmark_results.json"
    assert results_path.exists()
    written = json.loads(results_path.read_text())
    assert written["n_samples"] == 3
    assert len(written["records"]) == 3
    for key in ("compute_time_ms", "pixel_error"):
        assert key in written
    for time_key in ("mean", "median", "p95", "max"):
        assert time_key in written["compute_time_ms"]
        assert written["compute_time_ms"][time_key] >= 0.0

    example_pngs = list(tmp_path.glob("example_*.png"))
    assert len(example_pngs) == 1
