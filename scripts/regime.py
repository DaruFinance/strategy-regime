"""Latent regime classification on market data, then evaluate whether
conditioning strategy selection on regime improves out-of-sample
performance versus a regime-agnostic baseline.

Pipeline:
    1. Build per-bar features on OHLCV: log-returns, realized vol, ATR/price.
    2. Fit a Gaussian HMM with K hidden states (default K=3). Compare BIC for
       K in {2, 3, 4} when run with --bic-search.
    3. Decode the most likely regime sequence (Viterbi).
    4. For each strategy in the strategies/ Parquet, compute conditional
       OOS Sharpe per regime by joining trade entry/exit candle indices
       to the regime label of that bar.
    5. Compare two selection rules on the live-proxy windows:
         agnostic: top-K strategies by historical OOS-raw Sharpe
         conditional: top-K strategies for the *current* regime label

Usage:
    python scripts/regime.py                        # synthetic demo
    python scripts/regime.py --from-data \\
        --ohlcv /home/daru/quant-research-framework-rs/data/BTCUSDT_30m.csv \\
        --asset BTC_30m_27W
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

OUT_FIG = Path(__file__).resolve().parent.parent / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_FIG.parent / "regime.json"

# -------------------------------------------------------------------
# Feature builders
# -------------------------------------------------------------------

def ohlcv_features(df: pd.DataFrame, vol_window: int = 48) -> pd.DataFrame:
    """Per-bar features for the regime model.
    Inputs: df with columns time, open, high, low, close.
    Output: DataFrame indexed like the input, with feature columns and a NaN
    mask in the warmup window.
    """
    df = df.copy()
    df["logret"] = np.log(df["close"]).diff()
    df["abs_ret"] = df["logret"].abs()
    df["vol"] = df["abs_ret"].rolling(vol_window, min_periods=vol_window).mean()
    df["atr"] = ((df["high"] - df["low"]) / df["close"]
                 ).rolling(vol_window, min_periods=vol_window).mean()
    df["trend"] = df["logret"].rolling(vol_window, min_periods=vol_window).mean()
    df = df.dropna()
    return df


# -------------------------------------------------------------------
# HMM
# -------------------------------------------------------------------

def fit_hmm(X: np.ndarray, n_states: int = 3, n_iter: int = 60,
            seed: int = 42, cov_type: str = "diag"
            ) -> tuple[hmm.GaussianHMM, np.ndarray, float, float]:
    """Fit a Gaussian HMM with diagonal covariance (stable on collinear
    market features). Returns (model, decoded_states, loglik, bic)."""
    np.random.seed(seed)
    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type=cov_type,
                            n_iter=n_iter, random_state=seed,
                            tol=1e-3, init_params="stmc")
    model.fit(X)
    states = model.predict(X)
    ll = float(model.score(X))
    n, d = X.shape
    p = n_states - 1
    p += n_states * (n_states - 1)
    p += n_states * d                                       # means
    if cov_type == "full":
        p += n_states * (d * (d + 1) // 2)
    elif cov_type == "diag":
        p += n_states * d
    elif cov_type == "spherical":
        p += n_states
    elif cov_type == "tied":
        p += d * (d + 1) // 2
    bic = -2 * ll + p * np.log(n)
    return model, states, ll, bic


def reorder_states_by_vol(model, states: np.ndarray,
                          vol_idx: int = 1) -> np.ndarray:
    """Sort states by their mean of the volatility feature (ascending), so
    state-0 = lowest-vol, state-(K-1) = highest-vol. Returns relabelled
    state sequence; mapping is consistent across runs/assets given a fixed
    feature order.
    """
    means = model.means_[:, vol_idx]
    order = np.argsort(means)
    remap = {old: new for new, old in enumerate(order)}
    return np.vectorize(remap.get)(states)


# -------------------------------------------------------------------
# Synthetic demo (regime-switching returns)
# -------------------------------------------------------------------

def synthetic_returns(n_bars: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Three-regime synthetic series:
        low-vol up   (mu=+0.0005, sigma=0.005)
        high-vol     (mu=0,        sigma=0.020)
        low-vol down (mu=-0.0005, sigma=0.005)
    Markov transition with sticky persistence.
    """
    rng = np.random.default_rng(seed)
    P = np.array([
        [0.985, 0.010, 0.005],
        [0.015, 0.970, 0.015],
        [0.005, 0.010, 0.985],
    ])
    means = np.array([0.0005, 0.0, -0.0005])
    sds = np.array([0.005, 0.020, 0.005])
    state = 0
    states = np.zeros(n_bars, dtype=int)
    rets = np.zeros(n_bars)
    for t in range(n_bars):
        state = rng.choice(3, p=P[state])
        states[t] = state
        rets[t] = rng.normal(means[state], sds[state])
    close = 1000.0 * np.exp(np.cumsum(rets))
    high = close * np.exp(np.abs(rets) * 0.5 + 0.001)
    low = close * np.exp(-(np.abs(rets) * 0.5 + 0.001))
    open_ = np.r_[1000.0, close[:-1]]
    df = pd.DataFrame({
        "time": np.arange(n_bars) * 1800,
        "open": open_, "high": high, "low": low, "close": close,
    })
    df["true_state"] = states
    return df


# -------------------------------------------------------------------
# Strategy-side: conditional Sharpe per regime
# -------------------------------------------------------------------

def conditional_sharpe(trades: pd.DataFrame, regimes: np.ndarray
                       ) -> pd.DataFrame:
    """Per (strategy, regime) Sharpe of net trade PnLs (using exit-bar regime).
    trades: columns [strategy_name, exit_idx, pnl]
    regimes: 1d array indexed by bar (NaN allowed for warmup).
    Returns DataFrame [strategy_name, regime, n_trades, mean_pnl, sharpe].
    """
    valid = (trades["exit_idx"] >= 0) & (trades["exit_idx"] < len(regimes))
    sub = trades[valid].copy()
    sub["regime"] = regimes[sub["exit_idx"].to_numpy()]
    sub = sub.dropna(subset=["regime"])
    sub["regime"] = sub["regime"].astype(int)
    grp = sub.groupby(["strategy_name", "regime"])["pnl"]
    out = pd.DataFrame({
        "n_trades": grp.size(),
        "mean_pnl": grp.mean(),
        "std_pnl": grp.std(ddof=1),
    }).reset_index()
    out["sharpe"] = out["mean_pnl"] / out["std_pnl"].replace(0, np.nan) * np.sqrt(252)
    return out


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_regime_segmentation(df: pd.DataFrame, states: np.ndarray,
                             out_path: Path, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(df.index, df["close"], color="#222", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("close (log)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    K = int(states.max()) + 1
    palette = ["#4ccc7c", "#ccac4c", "#cc4c4c", "#4c8acc", "#9e4ccc"][:K]
    for s in range(K):
        m = states == s
        ax.fill_between(df.index, 0, 1, where=m, color=palette[s], alpha=0.7,
                        label=f"state {s}")
    ax.set_yticks([]); ax.set_xlabel("bar index")
    ax.legend(loc="upper right", fontsize=9, ncol=K)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conditional_vs_unconditional(per_strat: pd.DataFrame, out_path: Path):
    """Per regime, distribution of strategy Sharpe."""
    K = per_strat["regime"].max() + 1
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["#4ccc7c", "#ccac4c", "#cc4c4c", "#4c8acc", "#9e4ccc"][:K]
    for r in range(K):
        sub = per_strat[per_strat["regime"] == r]
        if sub.empty:
            continue
        ax.hist(sub["sharpe"].dropna(), bins=40, alpha=0.5, color=palette[r],
                label=f"regime {r} (n={len(sub):,}, "
                      f"profitable {(sub['sharpe'] > 0).mean()*100:.0f}%)")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Per-strategy conditional Sharpe (annualised)")
    ax.set_ylabel("count")
    ax.set_title("Conditional Sharpe distribution by regime")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_persistence(model, out_path: Path):
    P = model.transmat_
    K = P.shape[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(P, vmin=0, vmax=1, cmap="Reds")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    color="black" if P[i, j] < 0.6 else "white", fontsize=10)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title("Regime transition matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------

def run_hmm_pipeline(ohlcv: pd.DataFrame, n_states: int, label: str,
                    bic_search: bool = False) -> dict:
    feat = ohlcv_features(ohlcv)
    # logret + vol + trend are nearly orthogonal; atr is highly correlated
    # with vol so we drop it to avoid singular covariances.
    feature_cols = ["logret", "vol", "trend"]
    X = feat[feature_cols].to_numpy()
    # standardise (HMM is sensitive to feature scale)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    out: dict = {"label": label, "n_bars": int(len(X)), "features": feature_cols}

    if bic_search:
        sweep = []
        for K in (2, 3, 4):
            t0 = time.time()
            _, _, ll, bic = fit_hmm(X, n_states=K)
            sweep.append({"K": K, "loglik": ll, "bic": bic,
                          "t_fit": round(time.time()-t0, 1)})
            print(f"  K={K}: BIC={bic:,.0f} (loglik={ll:,.0f}, t={time.time()-t0:.1f}s)")
        out["bic_sweep"] = sweep
        # pick best K (lowest BIC)
        n_states = min(sweep, key=lambda r: r["bic"])["K"]
        print(f"  best K by BIC = {n_states}")

    t0 = time.time()
    model, raw_states, ll, bic = fit_hmm(X, n_states=n_states)
    print(f"[{label}] HMM K={n_states} fit in {time.time()-t0:.1f}s "
          f"(loglik={ll:,.0f}, BIC={bic:,.0f})")
    states = reorder_states_by_vol(model, raw_states,
                                   vol_idx=feature_cols.index("vol"))
    # Persistence: prob of staying
    diag = np.diag(model.transmat_)
    out.update({"n_states": n_states, "loglik": ll, "bic": bic,
                "stay_probs": diag.tolist(),
                "regime_means": model.means_.tolist(),
                "regime_share": [float((states == s).mean())
                                 for s in range(n_states)]})
    plot_regime_segmentation(feat.reset_index(drop=True), states,
                             OUT_FIG / f"fig_segmentation_{label}.png",
                             f"{label} regime segmentation (K={n_states})")
    plot_regime_persistence(model, OUT_FIG / f"fig_transitions_{label}.png")
    return out, feat, states


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv",
                    default="/home/daru/quant-research-framework-rs/data/BTCUSDT_30m.csv")
    ap.add_argument("--asset", default=None,
                    help="strategies/asset_dir to evaluate against (Phase 3 join)")
    ap.add_argument("-K", "--n-states", type=int, default=3)
    ap.add_argument("--bic-search", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="(rare) regime-switching synthetic returns; only for "
                         "testing the analysis machinery.")
    args = ap.parse_args()

    summary: dict = {"mode": "synthetic" if args.synthetic else "data"}
    if not args.synthetic:
        df = pd.read_csv(args.ohlcv)
        out, feat, states = run_hmm_pipeline(
            df, n_states=args.n_states,
            label=Path(args.ohlcv).stem,
            bic_search=args.bic_search)
    else:
        df = synthetic_returns()
        out, feat, states = run_hmm_pipeline(
            df, n_states=args.n_states, label="synthetic",
            bic_search=args.bic_search)
        # sanity: HMM should recover roughly the synthetic states' shares
        true_share = pd.Series(df["true_state"]).value_counts(normalize=True).sort_index().tolist()
        out["true_state_share"] = true_share
        # NMI between true and recovered (Hungarian-aligned)
        try:
            from sklearn.metrics import normalized_mutual_info_score
            true_aligned = df["true_state"].iloc[len(df) - len(states):].to_numpy()
            out["nmi_true_vs_decoded"] = float(
                normalized_mutual_info_score(true_aligned, states))
        except Exception:
            pass

    summary.update(out)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nresults summary -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
