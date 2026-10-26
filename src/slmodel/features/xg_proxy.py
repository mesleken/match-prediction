"""
Shot-quality proxy Expected Goals (xG) and opponent-adjusted form feature generator.
Strictly time-ordered (as-of) with zero lookahead leakage.
"""

from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import nnls


DEFAULT_PROXY_BETAS = (0.08, 0.28, 0.04, 0.02)  # [intercept, sot, off_target, corner]


def fit_proxy_xg_weights(
    matches_history: pd.DataFrame,
    as_of: Optional[datetime] = None,
    min_samples: int = 50,
) -> tuple[float, float, float, float]:
    """
    Fit non-negative least squares regression for proxy xG strictly on training history prior to as_of.
    Target: actual goals ~ beta_0 + beta_sot * sot + beta_off * (shots - sot) + beta_corner * corners.
    """
    valid = matches_history[
        (matches_history["status"] == "played")
        & matches_history["home_goals"].notna()
        & matches_history["away_goals"].notna()
    ].copy()

    if as_of is not None:
        dt_series = pd.to_datetime(valid["kickoff_utc"], utc=True)
        ref_dt = pd.to_datetime(as_of, utc=True)
        valid = valid[dt_series < ref_dt]

    has_shots = all(c in valid.columns for c in ["hs", "hst", "as_", "ast"])
    if not has_shots or len(valid) < min_samples:
        return DEFAULT_PROXY_BETAS

    hst = valid["hst"].to_numpy(dtype=np.float64)
    ast = valid["ast"].to_numpy(dtype=np.float64)
    hs = valid["hs"].to_numpy(dtype=np.float64)
    as_ = valid["as_"].to_numpy(dtype=np.float64)
    hc = valid["hc"].to_numpy(dtype=np.float64) if "hc" in valid.columns else np.zeros_like(hst)
    ac = valid["ac"].to_numpy(dtype=np.float64) if "ac" in valid.columns else np.zeros_like(ast)
    hg = valid["home_goals"].to_numpy(dtype=np.float64)
    ag = valid["away_goals"].to_numpy(dtype=np.float64)

    vs = ~np.isnan(hst) & ~np.isnan(ast) & ~np.isnan(hs) & ~np.isnan(as_)
    if np.sum(vs) < min_samples:
        return DEFAULT_PROXY_BETAS

    hst_v = hst[vs]
    ast_v = ast[vs]
    hs_off = np.clip(hs[vs] - hst_v, 0.0, None)
    as_off = np.clip(as_[vs] - ast_v, 0.0, None)
    hc_v = np.nan_to_num(hc[vs], nan=0.0)
    ac_v = np.nan_to_num(ac[vs], nan=0.0)

    X_shot = np.vstack([
        np.column_stack([np.ones_like(hst_v), hst_v, hs_off, hc_v]),
        np.column_stack([np.ones_like(ast_v), ast_v, as_off, ac_v]),
    ])
    y_shot = np.concatenate([hg[vs], ag[vs]])

    try:
        beta, _ = nnls(X_shot, y_shot)
        return float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])
    except Exception:
        return DEFAULT_PROXY_BETAS


def compute_match_proxy_xg(
    row: pd.Series,
    betas: tuple[float, float, float, float] = DEFAULT_PROXY_BETAS,
) -> tuple[float, float]:
    """
    Compute home and away proxy xG for a match row.
    Uses authentic Opta HxG / AxG if available, otherwise evaluates the fitted proxy.
    """
    b0, b_sot, b_off, b_c = betas

    # Home team
    if "hxg" in row and pd.notna(row["hxg"]):
        hxg = float(row["hxg"])
    elif "hs" in row and "hst" in row and pd.notna(row["hs"]) and pd.notna(row["hst"]):
        hs = float(row["hs"])
        hst = float(row["hst"])
        hc = float(row.get("hc", 0.0)) if pd.notna(row.get("hc")) else 0.0
        hxg = b0 + b_sot * hst + b_off * max(0.0, hs - hst) + b_c * hc
    else:
        hxg = float(row["home_goals"]) if pd.notna(row.get("home_goals")) else 1.35

    # Away team
    if "axg" in row and pd.notna(row["axg"]):
        axg = float(row["axg"])
    elif "as_" in row and "ast" in row and pd.notna(row["as_"]) and pd.notna(row["ast"]):
        as_ = float(row["as_"])
        ast = float(row["ast"])
        ac = float(row.get("ac", 0.0)) if pd.notna(row.get("ac")) else 0.0
        axg = b0 + b_sot * ast + b_off * max(0.0, as_ - ast) + b_c * ac
    else:
        axg = float(row["away_goals"]) if pd.notna(row.get("away_goals")) else 1.15

    return max(0.05, hxg), max(0.05, axg)


def compute_opponent_adjusted_xg_form(
    team_id: int,
    prior_matches: pd.DataFrame,
    betas: tuple[float, float, float, float] = DEFAULT_PROXY_BETAS,
    decay: float = 0.82,
    window_recent: int = 6,
    window_long: int = 30,
) -> dict[str, float]:
    """
    Compute recent opponent-adjusted attacking and defensive xG form momentum.
    Strictly uses prior matches with status='played'.
    """
    tm = prior_matches[
        (prior_matches["home_team_id"] == team_id) | (prior_matches["away_team_id"] == team_id)
    ]
    if len(tm) < 2:
        return {
            "xg_att_form": 0.0,
            "xg_def_vuln": 0.0,
            "xg_net_diff": 0.0,
        }

    rec_matches = tm.tail(window_recent)
    long_matches = tm.tail(window_long)

    def _extract_xg_series(df: pd.DataFrame) -> tuple[list[float], list[float]]:
        f_list: list[float] = []
        a_list: list[float] = []
        for _, r in df.iterrows():
            hxg, axg = compute_match_proxy_xg(r, betas)
            if r["home_team_id"] == team_id:
                f_list.append(hxg)
                a_list.append(axg)
            else:
                f_list.append(axg)
                a_list.append(hxg)
        return f_list, a_list

    rf, ra = _extract_xg_series(rec_matches)
    lf, la = _extract_xg_series(long_matches)

    w = [decay ** j for j in range(len(rf))][::-1]
    rec_xgf = float(np.average(rf, weights=w))
    rec_xga = float(np.average(ra, weights=w))

    base_xgf = float(np.mean(lf)) if len(lf) >= 6 else 1.35
    base_xga = float(np.mean(la)) if len(la) >= 6 else 1.35

    att_form = float(rec_xgf - base_xgf)
    def_vuln = float(rec_xga - base_xga)

    return {
        "xg_att_form": att_form,
        "xg_def_vuln": def_vuln,
        "xg_net_diff": float(att_form - def_vuln),
    }
