#!/usr/bin/env python3
"""
Predictive Monte Carlo simulation for a HYPOTHETICAL FIFA World Cup FINAL:
Argentina vs Spain (neutral venue).

This is a statistical sports-analytics exercise, NOT a prediction of a real
scheduled match. It fits a Dixon-Coles / Poisson attack-defence model to real
historical international results and simulates the final many times.

Data source: martj42/international_results (the open GitHub source behind the
Kaggle "International football results from 1872 to ..." dataset).

Usage:
    python3 simulate_final.py

Outputs a text summary and writes sim_output.json (consumed by the HTML report).
"""

import os
import json
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "results.csv")
SHOOTOUTS_CSV = os.path.join(HERE, "shootouts.csv")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

TEAM_A = "Argentina"
TEAM_B = "Spain"

RECENT_YEARS = 12          # only fit on the last N years (squads/strength drift)
HALF_LIFE_DAYS = 365 * 3   # exponential time-decay half life (recent games matter more)
RIDGE = 0.5                # L2 shrinkage on attack/defence params (stabilises weak teams)
MAX_GOALS = 12             # truncation for the scoreline probability grid
N_SIMS = 500_000           # Monte Carlo trials (justified below / in report)
SEED = 20260718


# ---------------------------------------------------------------------------
# Data loading (cached locally)
# ---------------------------------------------------------------------------
def _download(url, path):
    print(f"  downloading {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(path, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"ERROR: could not download real data from {url}\n"
            f"  reason: {exc}\n"
            "  Refusing to fabricate data. Please retrieve the CSV manually."
        )


def load_data():
    if not os.path.exists(RESULTS_CSV):
        print("results.csv not cached locally; fetching real data...")
        _download(RESULTS_URL, RESULTS_CSV)
    if not os.path.exists(SHOOTOUTS_CSV):
        _download(SHOOTOUTS_URL, SHOOTOUTS_CSV)

    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # neutral column is TRUE/FALSE strings or bools
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    shootouts = None
    if os.path.exists(SHOOTOUTS_CSV):
        shootouts = pd.read_csv(SHOOTOUTS_CSV)
    return df, shootouts


# ---------------------------------------------------------------------------
# Model fitting: weighted, ridge-regularised bivariate Poisson (attack/defence)
#
#   log(lambda_home) = mu + home_adv * (venue not neutral) + att[home] + def[away]
#   log(lambda_away) = mu +                                 att[away] + def[home]
#
# Fitted by minimising the weighted negative Poisson log-likelihood with a small
# L2 penalty on the team parameters (identifiability + shrinkage for teams with
# few games). A Dixon-Coles low-score correlation parameter rho is fitted
# afterwards on the same weighted data.
# ---------------------------------------------------------------------------
def build_design(df):
    cutoff = df["date"].max() - pd.Timedelta(days=365 * RECENT_YEARS)
    d = df[df["date"] >= cutoff].copy()

    # exponential time-decay weight
    age_days = (d["date"].max() - d["date"]).dt.days.to_numpy()
    lam_decay = np.log(2) / HALF_LIFE_DAYS
    d["w"] = np.exp(-lam_decay * age_days)

    teams = sorted(pd.unique(d[["home_team", "away_team"]].values.ravel()))
    tidx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # parameter layout: [mu, home_adv, att(n_teams), def(n_teams)]
    P = 2 + 2 * n_teams
    off_att = 2
    off_def = 2 + n_teams

    n = len(d)
    home = d["home_team"].map(tidx).to_numpy()
    away = d["away_team"].map(tidx).to_numpy()
    hs = d["home_score"].to_numpy(dtype=float)
    as_ = d["away_score"].to_numpy(dtype=float)
    not_neutral = (~d["neutral"].to_numpy()).astype(float)
    w = d["w"].to_numpy()

    # Two observation rows per match (home-goals row, away-goals row) -> 2n rows
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for i in range(n):
        # home-goals observation (row i)
        add(i, 0, 1.0)                         # mu
        add(i, 1, not_neutral[i])              # home advantage (only if not neutral)
        add(i, off_att + home[i], 1.0)         # attack of home team
        add(i, off_def + away[i], 1.0)         # defence of away team
        # away-goals observation (row n+i)
        add(n + i, 0, 1.0)                      # mu
        add(n + i, off_att + away[i], 1.0)      # attack of away team
        add(n + i, off_def + home[i], 1.0)      # defence of home team

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n, P))
    y = np.concatenate([hs, as_])
    wobs = np.concatenate([w, w])

    ridge_mask = np.ones(P)
    ridge_mask[0] = 0.0   # don't penalise intercept
    ridge_mask[1] = 0.0   # don't penalise home advantage

    meta = dict(
        teams=teams, tidx=tidx, off_att=off_att, off_def=off_def,
        n_teams=n_teams, P=P, d=d, home=home, away=away, hs=hs, as_=as_,
        not_neutral=not_neutral, w=w,
    )
    return X, y, wobs, ridge_mask, meta


def fit_poisson(X, y, wobs, ridge_mask):
    P = X.shape[1]

    def nll_and_grad(beta):
        eta = X.dot(beta)
        eta = np.clip(eta, -30, 30)
        lam = np.exp(eta)
        # weighted negative Poisson log-likelihood (drop constant log(y!))
        nll = np.sum(wobs * (lam - y * eta))
        nll += 0.5 * RIDGE * np.sum(ridge_mask * beta * beta)
        grad = X.T.dot(wobs * (lam - y))
        grad = np.asarray(grad).ravel() + RIDGE * ridge_mask * beta
        return nll, grad

    beta0 = np.zeros(P)
    beta0[0] = np.log(max(y.mean(), 0.1))
    res = minimize(nll_and_grad, beta0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500, ftol=1e-10))
    return res.x


def fit_rho(meta, beta):
    """Fit Dixon-Coles low-score correlation rho on weighted recent data,
    holding the Poisson lambdas from the fitted model fixed."""
    off_att, off_def = meta["off_att"], meta["off_def"]
    mu, home_adv = beta[0], beta[1]
    home, away = meta["home"], meta["away"]
    hs, as_ = meta["hs"].astype(int), meta["as_"].astype(int)
    not_neutral, w = meta["not_neutral"], meta["w"]

    att = beta[off_att:off_att + meta["n_teams"]]
    dfn = beta[off_def:off_def + meta["n_teams"]]

    lh = np.exp(mu + home_adv * not_neutral + att[home] + dfn[away])
    la = np.exp(mu + att[away] + dfn[home])

    # only rows where the DC correction is active (goals in {0,1})
    active = (hs <= 1) & (as_ <= 1)

    def neg_ll(rho):
        tau = np.ones(len(hs))
        m00 = active & (hs == 0) & (as_ == 0)
        m01 = active & (hs == 0) & (as_ == 1)
        m10 = active & (hs == 1) & (as_ == 0)
        m11 = active & (hs == 1) & (as_ == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        if np.any(tau <= 0):
            return 1e12
        return -np.sum(w * np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


def team_rates(meta, beta, team_att, team_def_opp):
    """Neutral-venue expected goals for a team given its attack and the
    opponent's defence (home advantage removed -> neutral final)."""
    mu = beta[0]
    return np.exp(mu + team_att + team_def_opp)


# ---------------------------------------------------------------------------
# Dixon-Coles joint scoreline distribution (for regulation sampling)
# ---------------------------------------------------------------------------
def poisson_pmf_vec(lam, kmax):
    k = np.arange(kmax + 1)
    logp = -lam + k * np.log(lam) - np.array([np.sum(np.log(np.arange(1, i + 1))) for i in k])
    return np.exp(logp)


def dc_joint_matrix(lam_a, lam_b, rho, kmax):
    pa = poisson_pmf_vec(lam_a, kmax)
    pb = poisson_pmf_vec(lam_b, kmax)
    M = np.outer(pa, pb)
    # Dixon-Coles tau correction on the four low-score cells
    M[0, 0] *= 1 - lam_a * lam_b * rho
    M[0, 1] *= 1 + lam_a * rho
    M[1, 0] *= 1 + lam_b * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    M /= M.sum()
    return M


# ---------------------------------------------------------------------------
# Monte Carlo simulation of the final
# ---------------------------------------------------------------------------
def simulate(lam_a_reg, lam_b_reg, rho, n_sims, pen_p_a=0.5, seed=SEED):
    """
    lam_a_reg, lam_b_reg : expected regulation (90') goals for A and B.
    Extra time: 30' at 1/3 of the full-match rate.
    Shootout: Bernoulli(pen_p_a) that A wins.
    Regulation goals are drawn from the fitted Dixon-Coles joint distribution
    (captures low-score correlation); extra time uses independent Poisson.
    """
    rng = np.random.default_rng(seed)
    kmax = MAX_GOALS

    # sample regulation scoreline from the DC joint pmf
    M = dc_joint_matrix(lam_a_reg, lam_b_reg, rho, kmax)
    flat = M.ravel()
    flat = flat / flat.sum()
    draws = rng.choice(flat.size, size=n_sims, p=flat)
    ga = (draws // (kmax + 1)).astype(np.int16)
    gb = (draws % (kmax + 1)).astype(np.int16)

    a_reg_win = ga > gb
    b_reg_win = gb > ga
    tied = ga == gb

    # extra time for tied matches (30 min ~ 1/3 of a 90-min rate)
    et_a = rng.poisson(lam_a_reg / 3.0, size=n_sims).astype(np.int16)
    et_b = rng.poisson(lam_b_reg / 3.0, size=n_sims).astype(np.int16)
    et_a = np.where(tied, et_a, 0)
    et_b = np.where(tied, et_b, 0)
    ta = ga + et_a
    tb = gb + et_b

    a_et_win = tied & (ta > tb)
    b_et_win = tied & (tb > ta)
    still_tied = tied & (ta == tb)

    # penalty shootout
    pens = rng.random(n_sims) < pen_p_a
    a_pen_win = still_tied & pens
    b_pen_win = still_tied & ~pens

    a_win = a_reg_win | a_et_win | a_pen_win
    b_win = b_reg_win | b_et_win | b_pen_win

    res = dict(
        n=n_sims,
        a_win=float(a_win.mean()),
        b_win=float(b_win.mean()),
        a_reg_win=float(a_reg_win.mean()),
        b_reg_win=float(b_reg_win.mean()),
        draw_regulation=float(tied.mean()),
        a_et_win=float(a_et_win.mean()),
        b_et_win=float(b_et_win.mean()),
        extra_time_decided=float((a_et_win | b_et_win).mean()),
        shootout=float(still_tied.mean()),
        a_pen_win=float(a_pen_win.mean()),
        b_pen_win=float(b_pen_win.mean()),
        exp_goals_a=float(ga.mean()),
        exp_goals_b=float(gb.mean()),
    )

    # most likely regulation scorelines (cap display at 5-5)
    cap = 6
    grid = np.zeros((cap, cap))
    for i in range(cap):
        for j in range(cap):
            grid[i, j] = np.mean((ga == i) & (gb == j))
    scorelines = []
    for i in range(cap):
        for j in range(cap):
            scorelines.append((f"{i}-{j}", i, j, float(grid[i, j])))
    scorelines.sort(key=lambda x: -x[3])
    res["scoreline_grid"] = grid.tolist()
    res["top_scorelines"] = scorelines[:12]
    return res


def convergence_check(lam_a, lam_b, rho, n_full):
    small = simulate(lam_a, lam_b, rho, n_full // 10, seed=SEED + 1)
    full = simulate(lam_a, lam_b, rho, n_full, seed=SEED + 2)
    return small, full


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("HYPOTHETICAL WORLD CUP FINAL SIMULATION:", TEAM_A, "vs", TEAM_B)
    print("(statistical what-if model, not a prediction of a real match)")
    print("=" * 70)

    print("\n[1/5] Loading real historical data...")
    df, shootouts = load_data()
    print(f"  {len(df):,} matches with scores, {df.date.min().date()} -> {df.date.max().date()}")

    print("\n[2/5] Building design matrix (last %d years, time-decayed)..." % RECENT_YEARS)
    X, y, wobs, ridge_mask, meta = build_design(df)
    print(f"  {len(meta['d']):,} matches used, {meta['n_teams']} teams, {meta['P']} parameters")

    print("\n[3/5] Fitting weighted ridge Poisson attack/defence model...")
    beta = fit_poisson(X, y, wobs, ridge_mask)
    rho = fit_rho(meta, beta)
    mu, home_adv = beta[0], beta[1]
    print(f"  intercept mu={mu:.3f}  home_adv={home_adv:.3f} "
          f"(exp={np.exp(home_adv):.3f}x)  Dixon-Coles rho={rho:.4f}")

    tidx = meta["tidx"]
    if TEAM_A not in tidx or TEAM_B not in tidx:
        raise SystemExit(f"ERROR: {TEAM_A} or {TEAM_B} not present in recent data.")
    ia, ib = tidx[TEAM_A], tidx[TEAM_B]
    att = beta[meta["off_att"]:meta["off_att"] + meta["n_teams"]]
    dfn = beta[meta["off_def"]:meta["off_def"] + meta["n_teams"]]

    # neutral-venue expected regulation goals (no home advantage)
    lam_a = team_rates(meta, beta, att[ia], dfn[ib])
    lam_b = team_rates(meta, beta, att[ib], dfn[ia])
    print(f"  {TEAM_A}: attack={att[ia]:+.3f} defence={dfn[ia]:+.3f}")
    print(f"  {TEAM_B}: attack={att[ib]:+.3f} defence={dfn[ib]:+.3f}")
    print(f"  Neutral-venue expected regulation goals -> "
          f"{TEAM_A} {lam_a:.3f} | {TEAM_B} {lam_b:.3f}")

    # optional small penalty-shootout tendency from historical shootout record
    pen_p_a = 0.5
    pen_note = "50/50 baseline (no strong data signal)"
    if shootouts is not None:
        def rec(team):
            m = (shootouts.home_team == team) | (shootouts.away_team == team)
            wins = (shootouts.winner == team).sum()
            return int(wins), int(m.sum())
        wa, na = rec(TEAM_A)
        wb, nb = rec(TEAM_B)
        # shrink strongly toward 0.5 (small samples); minor adjustment only
        ra = (wa + 5) / (na + 10) if na else 0.5
        rb = (wb + 5) / (nb + 10) if nb else 0.5
        pen_p_a = 0.5 + 0.5 * (ra - rb)  # relative edge, damped
        pen_p_a = float(np.clip(pen_p_a, 0.42, 0.58))
        pen_note = (f"{TEAM_A} {wa}/{na} vs {TEAM_B} {wb}/{nb} historical shootouts "
                    f"-> damped P({TEAM_A})={pen_p_a:.3f}")
    print(f"  Penalty model: {pen_note}")

    print(f"\n[4/5] Convergence check (N/10 vs N)...")
    small, full = convergence_check(lam_a, lam_b, rho, N_SIMS)
    se = (full["a_win"] * (1 - full["a_win"]) / N_SIMS) ** 0.5
    print(f"  {TEAM_A} win: N/10={small['a_win']*100:.2f}%  "
          f"N={full['a_win']*100:.2f}%  (95% CI +/-{1.96*se*100:.2f}%)")
    print(f"  shootout: N/10={small['shootout']*100:.2f}%  N={full['shootout']*100:.2f}%")

    print(f"\n[5/5] Running Monte Carlo simulation (N={N_SIMS:,})...")
    res = simulate(lam_a, lam_b, rho, N_SIMS, pen_p_a=pen_p_a)

    # ---- report ----
    print("\n" + "=" * 70)
    print("RESULTS  (N = {:,} simulations)".format(N_SIMS))
    print("=" * 70)
    print(f"\nOVERALL WIN PROBABILITY")
    print(f"  {TEAM_A:10s}: {res['a_win']*100:5.2f}%")
    print(f"  {TEAM_B:10s}: {res['b_win']*100:5.2f}%")
    print(f"\nHOW THE FINAL IS DECIDED")
    print(f"  Decided in regulation (90'): {(res['a_reg_win']+res['b_reg_win'])*100:5.2f}%"
          f"   [{TEAM_A} {res['a_reg_win']*100:.2f}% / {TEAM_B} {res['b_reg_win']*100:.2f}%]")
    print(f"  Level after 90' (goes to ET): {res['draw_regulation']*100:5.2f}%")
    print(f"  Decided in extra time:        {res['extra_time_decided']*100:5.2f}%"
          f"   [{TEAM_A} {res['a_et_win']*100:.2f}% / {TEAM_B} {res['b_et_win']*100:.2f}%]")
    print(f"  Goes to penalty shootout:     {res['shootout']*100:5.2f}%"
          f"   [{TEAM_A} {res['a_pen_win']*100:.2f}% / {TEAM_B} {res['b_pen_win']*100:.2f}%]")
    print(f"\nEXPECTED REGULATION GOALS")
    print(f"  {TEAM_A}: {res['exp_goals_a']:.2f}   {TEAM_B}: {res['exp_goals_b']:.2f}")
    print(f"\nMOST LIKELY REGULATION SCORELINES ({TEAM_A}-{TEAM_B})")
    for name, i, j, p in res["top_scorelines"][:8]:
        print(f"  {name}: {p*100:5.2f}%")

    # ---- persist for the HTML report ----
    out = dict(
        team_a=TEAM_A, team_b=TEAM_B,
        data_source="martj42/international_results (GitHub, open source of the Kaggle dataset)",
        n_matches=int(len(df)),
        date_min=str(df.date.min().date()), date_max=str(df.date.max().date()),
        n_matches_fit=int(len(meta["d"])), n_teams=int(meta["n_teams"]),
        recent_years=RECENT_YEARS, half_life_days=HALF_LIFE_DAYS,
        mu=float(mu), home_adv=float(home_adv), rho=float(rho),
        lam_a=float(lam_a), lam_b=float(lam_b),
        att_a=float(att[ia]), def_a=float(dfn[ia]),
        att_b=float(att[ib]), def_b=float(dfn[ib]),
        pen_p_a=float(pen_p_a), pen_note=pen_note,
        n_sims=N_SIMS,
        convergence=dict(a_win_small=small["a_win"], a_win_full=full["a_win"],
                         shootout_small=small["shootout"], shootout_full=full["shootout"]),
        results=res,
    )
    with open(os.path.join(HERE, "sim_output.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote sim_output.json")
    print("Done.")


if __name__ == "__main__":
    main()
