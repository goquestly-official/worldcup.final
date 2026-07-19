# World Cup Final: Argentina vs Spain — Monte Carlo Model

A statistical sports-analytics exercise, **not** a prediction of any real scheduled
match. It fits a Dixon-Coles / Poisson attack-defence model to real historical
international results and simulates a hypothetical neutral-venue World Cup final
between Argentina and Spain 500,000 times, including extra time and penalty shootouts.

## Data

Real historical match data (49,518 matches, 1872–2026) from
[martj42/international_results](https://github.com/martj42/international_results),
the open GitHub source behind the Kaggle dataset *"International football results
from 1872 to ..."*. The model is fit on the most recent 12 years (11,577 matches),
with a 3-year exponential time-decay weight so recent form dominates.

## Model

- Weighted, ridge-regularised **Poisson attack/defence ratings** per team, with a
  **Dixon-Coles low-score correction** (rho ≈ -0.05).
- Neutral-venue expected goals: **Argentina 1.03**, **Spain 1.07** (home-advantage
  term removed since a final is neutral).
- Regulation (90') scorelines sampled from the fitted joint distribution.
- Level after 90' → extra time (30') simulated as an additional Poisson period at
  ~1/3 the full-match scoring rate.
- Still level → penalty shootout, modeled as a near-coin-flip damped by each team's
  historical shootout record (Argentina 15/23, Spain 7/14 historically).

## Result

| | Argentina | Spain |
|---|---|---|
| Win probability | 49.7% | 50.3% |
| — in regulation | 33.3% | 35.4% |
| — in extra time | 6.8% | 7.1% |
| — on penalties | 9.6% | 7.9% |

Effectively a coin flip — the ~0.6pt gap is inside simulation noise (95% CI ≈ ±0.14%
at N=500,000). 31.4% of simulated finals reach extra time; 17.5% go to penalties.
Most likely regulation scorelines: **1-1** (14.3%), 0-0 (12.9%), 0-1 (12.4%),
1-0 (11.9%).

## Files

- `simulate_final.py` — reproducible end-to-end script: fetches/caches data, fits
  the model, runs the simulation, writes `sim_output.json`.
- `sim_output.json` — full simulation output, including the complete 6×6 regulation
  scoreline probability grid.
- `report.html` — visual report (win probability breakdown, scoreline heatmap,
  methodology).
- `results.csv` / `shootouts.csv` — cached historical data snapshot used for this run.

## Usage

```bash
pip3 install --user pandas numpy scipy
python3 simulate_final.py
```
