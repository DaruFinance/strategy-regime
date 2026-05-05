# strategy-regime

**Latent regime classification on market data, then conditional strategy
selection.**

> Companion repository to the M-series of reference implementations on
> [daniel-v-gatto.com](https://daniel-v-gatto.com). Market data is the
> OHLCV feed used by
> [`quant-research-framework-rs`](https://github.com/DaruFinance/quant-research-framework-rs);
> downstream strategy joins use the `pnl_daily/` Parquet substrate produced by
> [`strategy-pnl-daily-rs`](https://github.com/DaruFinance/strategy-pnl-daily-rs)
> (planned).

## What this is

A walk-forward strategy population is sensitive to which market regime
the OOS period happens to fall in. To make that explicit we fit a
Gaussian HMM directly on per-bar market features (returns, realised
volatility, trend slope), decode the most likely regime sequence with
Viterbi, and use the labels as a conditioning variable.

Two questions:

1. Does selecting strategies *conditional on the current regime label*
   beat selecting strategies regime-agnostic, on the live-proxy windows?
2. How sticky are the regimes — i.e., is the label still accurate one
   or two windows after detection?

This repo answers (2) directly via the transition matrix's stay-probabilities.
(1) requires joining the regime sequence to the strategy trade exits and
running the agnostic-vs-conditional comparison; that pipeline is wired
but the headline number for it is in the next release.

## Reproduce

```bash
git clone https://github.com/DaruFinance/strategy-regime
cd strategy-regime
pip install -e .
python scripts/regime.py --bic-search       # default: BTC OHLCV
```

The default reads `BTCUSDT_30m.csv` from the production framework's
`data/` dir and writes `figures/fig_segmentation_BTCUSDT_30m.png`,
`figures/fig_transitions_BTCUSDT_30m.png`, plus `regime.json`. Pass
`--ohlcv path/to/file.csv` to point elsewhere.

A `--synthetic` flag runs the regime-switching toy returns demo (only
used by the smoke test).

## Problem statement

Define a feature map on each bar `t`:

```
f_t = [ logret_t, vol_t, trend_t ]
```

where `vol_t` is the rolling mean absolute return and `trend_t` the rolling
mean log-return over a 24-hour window. Fit a Gaussian HMM with `K` hidden
states and diagonal covariances:

```
s_t ~ HMM(K)
f_t | s_t = k ~ N(μ_k, diag(σ_k²))
```

Choose `K` by BIC over `{2, 3, 4}` (use `--bic-search`). Diagonal
covariance avoids the singular-matrix failure that full covariance hits
when `vol` and `atr` are nearly collinear.

For each strategy `s` and regime label `r`, the conditional Sharpe is the
Sharpe ratio of trades whose *exit bar* falls under regime `r`. The
selection comparison runs two rules on the live-proxy windows:

- **agnostic**: pick top-`K` strategies by all-window OOS-raw Sharpe.
- **conditional**: pick top-`K` strategies for the current regime label
  (using only history-window data to fit the regime model).

## Headline result

**Across all 10 deepest-WFO crypto assets, BIC unanimously selects K=4
regimes.** BTC, ETH, LTC, TRX, XRP, LINK, ZEC, DOGE, BCH, AVAX — each one
sweeps {2, 3, 4} and lands on K=4 every time. The same regime-count
appears to be a near-universal feature of crypto market structure at the
30m timeframe.

Per-asset BIC sweep (lower is better; bold = selected):

| Asset | K=2 BIC | K=3 BIC | K=4 BIC |
|---|---:|---:|---:|
| BTC_30m_27W | 848,957 | 692,709 | **624,839** |
| ETH_30m_28W | 1,574,094 | 1,490,247 | **1,207,781** |
| LTC_30m_27W | 1,632,826 | 1,366,272 | **1,251,108** |
| TRX_30m_25W | 1,304,766 | 980,565 | **947,397** |
| XRP_30M_25W_new | 1,387,731 | 1,109,510 | **935,456** |
| LINK_30M_23W_new | 1,678,754 | 1,176,399 | **1,176,190** |
| ZEC_30m_22W | 1,264,025 | 1,175,470 | **979,321** |
| DOGE_30m_21W | 568,775 | 429,363 | **348,394** |
| BCH_30m_20W | 1,213,686 | 1,015,285 | **889,023** |
| AVAX_30m_17W | 1,026,748 | 858,940 | **780,651** |

For BTCUSDT_30m specifically (146,120 bars, ~9 years), the K=4 regimes
are interpretable (means in standardised feature space):

| state | logret | vol | trend | share | stay-prob | label |
|------:|------:|----:|------:|------:|----------:|:------|
| 0 | +0.06 | −0.08 | +0.69 | 33% | 0.969 | **bull trend** |
| 1 | +0.00 | −0.70 | +0.01 | 25% | 0.985 | **calm/range** |
| 2 | −0.05 | −0.12 | −0.56 | 24% | 0.970 | **bear trend** |
| 3 | −0.01 | +1.53 | −0.17 | 18% | 0.977 | **high-vol** |

All four regimes are sticky (stay-prob ≥ 0.97), so the decoded label is a
useful conditioning variable for strategy selection. Per-asset stay-prob
matrices are saved in `figures/fig_transitions_<asset>.png`.

## Usage

```bash
# Default (BTC 30m, BIC sweep over K in {2,3,4}):
python scripts/regime.py --bic-search

# Other asset:
python scripts/regime.py --ohlcv /home/daru/quant-research-framework-rs/data/SOLUSDT_1h.csv

# Fix K instead of searching:
python scripts/regime.py -K 3
```

`--asset BTC_30m_27W` (when implemented in the next release) will join the
decoded regimes to the trades/ Parquet and run the
agnostic-vs-conditional selection comparison on the live-proxy windows of
the corresponding strategies.

## References

- Hamilton, J. D. (1989). *A new approach to the economic analysis of
  nonstationary time series and the business cycle.* Econometrica.
- Ang, A. & Bekaert, G. (2002). *Regime switches in interest rates.* JBES.
- Rabiner, L. R. (1989). *A tutorial on hidden Markov models.* Proc. IEEE.

## License

MIT © Daniel Vieira Gatto.
