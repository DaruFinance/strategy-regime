"""Cross-asset BIC-sweep figure: shows K=4 winning across all 10 assets."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hard-coded from the BIC sweeps already run; future versions could write a JSON.
data = {
    "BTC_30m_27W":      {2: 848_957,    3: 692_709,    4: 624_839},
    "ETH_30m_28W":      {2: 1_574_094,  3: 1_490_247,  4: 1_207_781},
    "LTC_30m_27W":      {2: 1_632_826,  3: 1_366_272,  4: 1_251_108},
    "TRX_30m_25W":      {2: 1_304_766,  3: 980_565,    4: 947_397},
    "XRP_30M_25W_new":  {2: 1_387_731,  3: 1_109_510,  4: 935_456},
    "LINK_30M_23W_new": {2: 1_678_754,  3: 1_176_399,  4: 1_176_190},
    "ZEC_30m_22W":      {2: 1_264_025,  3: 1_175_470,  4: 979_321},
    "DOGE_30m_21W":     {2: 568_775,    3: 429_363,    4: 348_394},
    "BCH_30m_20W":      {2: 1_213_686,  3: 1_015_285,  4: 889_023},
    "AVAX_30m_17W":     {2: 1_026_748,  3: 858_940,    4: 780_651},
}

# Normalise each asset to its K=2 baseline so all curves fit on one axis.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
palette = plt.colormaps["tab10"]
assets = sorted(data.keys())

for i, a in enumerate(assets):
    s = data[a]
    ks = sorted(s.keys())
    bics = [s[k] for k in ks]
    base = bics[0]
    rel = [b / base for b in bics]
    axes[0].plot(ks, bics, "o-", color=palette(i % 10), linewidth=1.5, label=a)
    axes[1].plot(ks, rel, "o-", color=palette(i % 10), linewidth=1.5, label=a)

for ax in axes:
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("K (number of HMM hidden states)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

axes[0].set_ylabel("BIC (lower = better)")
axes[0].set_title("Per-asset BIC by K")
axes[0].set_yscale("log")

axes[1].set_ylabel("BIC normalised to K=2 baseline")
axes[1].set_title("BIC reduction relative to K=2 — every asset wins at K=4")
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.6)

fig.suptitle("Cross-asset BIC sweep: every asset selects K=4",
             fontsize=12)
fig.tight_layout()
out = ROOT / "figures" / "fig_bic_cross_asset.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
