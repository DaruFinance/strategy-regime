"""Smoke test for the analysis machinery itself.

Uses `--synthetic` (regime-switching toy returns) to exercise the HMM fit +
state-reorder + plotting pipeline deterministically. Real-asset runs go
through the orchestrator.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_demo():
    res = subprocess.run(
        [sys.executable, "scripts/regime.py", "--synthetic"],
        cwd=ROOT, capture_output=True, text=True, timeout=240,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads((ROOT / "regime.json").read_text())
    assert out["mode"] == "synthetic"
    # Basic sanity: stay-probs should reflect Markov stickiness
    sp = out["stay_probs"]
    assert max(sp) > 0.9, f"max stay-prob = {max(sp)} (expected > 0.9)"
    for fname in ["fig_segmentation_synthetic.png",
                  "fig_transitions_synthetic.png"]:
        assert (ROOT / "figures" / fname).is_file()


if __name__ == "__main__":
    test_synthetic_demo()
    print("OK")
