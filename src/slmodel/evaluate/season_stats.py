"""Season-level team standings, xG/xPts analytics, and match statistics engine (with 2026/27 xG support)."""

from typing import Any
import duckdb
import numpy as np
import pandas as pd
from slmodel.simulate.score_matrix import score_matrix
from slmodel.simulate.markets import one_x_two


def format_season_label(season_code: str) -> str:
    """Convert 4-digit season code like '2627' to '2026/27'."""
    s = str(season_code).strip()
    if len(s) == 4 and s.isdigit():
        return f"20{s[:2]}/{s[2:]}"
    return s


def _match_expected_points(xg_h: float, xg_a: float, rho: float = -0.05) -> tuple[float, float]:
    """Compute expected points (xPts) for home and away teams from match xG using Dixon-Coles matrix."""
    lam_h = max(0.05, float(xg_h))
    lam_a = max(0.05, float(xg_a))
    M = score_matrix(lam_h, lam_a, rho=rho)
    p1, px, p2 = one_x_two(M)
    xpts_h = 3.0 * float(p1) + 1.0 * float(px)
    xpts_a = 3.0 * float(p2) + 1.0 * float(px)
    return xpts_h, xpts_a


def compute_season_statistics(
    db_path: str = "data/warehouse.duckdb",
    season: str = "2627",
) -> dict[str, Any]:
    """
    Compute comprehensive team standings, xG/xGA/xGD/xPts metrics, shot/corner/card averages,
    and league-wide aggregates for a given Süper Lig season (defaults to '2627' = 2026/27).
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        q = """
            SELECT
                m.match_id, m.season, m.matchday, m.kickoff_utc,
                m.home_team_id, th.name AS home_team,
                m.away_team_id, ta.name AS away_team,
                m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
                m.status, m.crowd, m.is_derby,
                sh.shots AS hs, sh.shots_on_target AS hst, sh.corners AS hc,
                sh.fouls AS hf, sh.yellow AS hy, sh.red AS hr, sh.xg AS hxg,
                sa.shots AS as_, sa.shots_on_target AS ast, sa.corners AS ac,
                sa.fouls AS af, sa.yellow AS ay, sa.red AS ar, sa.xg AS axg
            FROM matches m
            JOIN teams th ON m.home_team_id = th.team_id
            JOIN teams ta ON m.away_team_id = ta.team_id
            LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
            LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
            WHERE m.season = ? AND m.status = 'played'
              AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
            ORDER BY m.kickoff_utc ASC, m.match_id ASC
        """
        df = con.execute(q, [str(season)]).fetchdf()
    finally:
        con.close()

    if df.empty:
        return {
            "season": str(season),
            "season_label": format_season_label(season),
            "matches_played": 0,
            "standings": pd.DataFrame(),
            "matches": pd.DataFrame(),
        }

    # Check if true xG is populated; if not, estimate proxy xG from shots on target, off-target shots, corners
    has_true_xg = bool(df["hxg"].notna().any() and df["axg"].notna().any())

    team_acc: dict[int, dict[str, Any]] = {}

    def _init_team(tid: int, tname: str) -> dict[str, Any]:
        if tid not in team_acc:
            team_acc[tid] = {
                "team_id": tid,
                "Takım": tname,
                "OM": 0,
                "G": 0,
                "B": 0,
                "M": 0,
                "AG": 0,
                "YG": 0,
                "Puan": 0,
                "xG": 0.0,
                "xGA": 0.0,
                "xPts": 0.0,
                "shots_for": 0,
                "shots_against": 0,
                "sot_for": 0,
                "sot_against": 0,
                "corners_for": 0,
                "corners_against": 0,
                "fouls_for": 0,
                "yellow_for": 0,
                "red_for": 0,
                "btts_count": 0,
                "over25_count": 0,
            }
        return team_acc[tid]

    match_display_rows = []

    for _, r in df.iterrows():
        hid = int(r["home_team_id"])
        aid = int(r["away_team_id"])
        hname = str(r["home_team"])
        aname = str(r["away_team"])
        hg = int(r["home_goals"])
        ag = int(r["away_goals"])
        hthg = int(r["ht_home_goals"]) if pd.notna(r["ht_home_goals"]) else 0
        htag = int(r["ht_away_goals"]) if pd.notna(r["ht_away_goals"]) else 0

        hs = int(r["hs"]) if pd.notna(r["hs"]) else 0
        as_ = int(r["as_"]) if pd.notna(r["as_"]) else 0
        hst = int(r["hst"]) if pd.notna(r["hst"]) else 0
        ast = int(r["ast"]) if pd.notna(r["ast"]) else 0
        hc = int(r["hc"]) if pd.notna(r["hc"]) else 0
        ac = int(r["ac"]) if pd.notna(r["ac"]) else 0
        hf = int(r["hf"]) if pd.notna(r["hf"]) else 0
        af = int(r["af"]) if pd.notna(r["af"]) else 0
        hy = int(r["hy"]) if pd.notna(r["hy"]) else 0
        ay = int(r["ay"]) if pd.notna(r["ay"]) else 0
        hr = int(r["hr"]) if pd.notna(r["hr"]) else 0
        ar = int(r["ar"]) if pd.notna(r["ar"]) else 0

        if pd.notna(r["hxg"]) and pd.notna(r["axg"]):
            hxg = float(r["hxg"])
            axg = float(r["axg"])
        else:
            # Shot-quality proxy xG when historical season lacks raw HxG/AxG
            hxg = 0.12 + 0.26 * hst + 0.04 * max(0, hs - hst) + 0.02 * hc
            axg = 0.12 + 0.26 * ast + 0.04 * max(0, as_ - ast) + 0.02 * ac

        xpts_h, xpts_a = _match_expected_points(hxg, axg)

        th = _init_team(hid, hname)
        ta = _init_team(aid, aname)

        th["OM"] += 1
        ta["OM"] += 1
        th["AG"] += hg
        th["YG"] += ag
        ta["AG"] += ag
        ta["YG"] += hg

        if hg > ag:
            th["G"] += 1
            th["Puan"] += 3
            ta["M"] += 1
        elif hg == ag:
            th["B"] += 1
            ta["B"] += 1
            th["Puan"] += 1
            ta["Puan"] += 1
        else:
            ta["G"] += 1
            ta["Puan"] += 3
            th["M"] += 1

        th["xG"] += hxg
        th["xGA"] += axg
        th["xPts"] += xpts_h

        ta["xG"] += axg
        ta["xGA"] += hxg
        ta["xPts"] += xpts_a

        th["shots_for"] += hs
        th["shots_against"] += as_
        th["sot_for"] += hst
        th["sot_against"] += ast
        th["corners_for"] += hc
        th["corners_against"] += ac
        th["fouls_for"] += hf
        th["yellow_for"] += hy
        th["red_for"] += hr

        ta["shots_for"] += as_
        ta["shots_against"] += hs
        ta["sot_for"] += ast
        ta["sot_against"] += hst
        ta["corners_for"] += ac
        ta["corners_against"] += hc
        ta["fouls_for"] += af
        ta["yellow_for"] += ay
        ta["red_for"] += ar

        is_btts = int(hg > 0 and ag > 0)
        is_o25 = int((hg + ag) > 2.5)
        th["btts_count"] += is_btts
        ta["btts_count"] += is_btts
        th["over25_count"] += is_o25
        ta["over25_count"] += is_o25

        dt_str = pd.to_datetime(r["kickoff_utc"]).strftime("%Y-%m-%d %H:%M")
        match_display_rows.append({
            "Hafta": int(r["matchday"]) if pd.notna(r["matchday"]) else None,
            "Tarih": dt_str,
            "Ev Sahibi": hname,
            "Skor": f"{hg} - {ag}",
            "İY": f"{hthg} - {htag}",
            "Deplasman": aname,
            "xG (Ev - Dep)": f"{hxg:.2f} - {axg:.2f}",
            "Şut (İsabetli)": f"{hs}({hst}) - {as_}({ast})",
            "Korner": f"{hc} - {ac}",
            "Kart (S/K)": f"{hy}/{hr} - {ay}/{ar}",
        })

    rows = []
    for tid, d in team_acc.items():
        om = max(1, d["OM"])
        av = d["AG"] - d["YG"]
        xgd = d["xG"] - d["xGA"]
        g_minus_xg = d["AG"] - d["xG"]
        pts_minus_xpts = d["Puan"] - d["xPts"]
        sot_pct = (d["sot_for"] / d["shots_for"] * 100.0) if d["shots_for"] > 0 else 0.0

        rows.append({
            "team_id": tid,
            "Takım": d["Takım"],
            "OM": d["OM"],
            "G": d["G"],
            "B": d["B"],
            "M": d["M"],
            "AG": d["AG"],
            "YG": d["YG"],
            "AV": av,
            "Puan": d["Puan"],
            "xPts": d["xPts"],
            "Puan - xPts": pts_minus_xpts,
            "xG": d["xG"],
            "xGA": d["xGA"],
            "xGD": xgd,
            "G - xG": g_minus_xg,
            "xG/M": d["xG"] / om,
            "xGA/M": d["xGA"] / om,
            "Şut/M": d["shots_for"] / om,
            "İsabetli Şut/M": d["sot_for"] / om,
            "Şut İsabet %": sot_pct,
            "Korner/M": d["corners_for"] / om,
            "Y.Korner/M": d["corners_against"] / om,
            "Faul/M": d["fouls_for"] / om,
            "Sarı Kart/M": d["yellow_for"] / om,
            "Kırmızı Kart": d["red_for"],
            "Üst 2.5 %": d["over25_count"] / om * 100.0,
            "KG Var %": d["btts_count"] / om * 100.0,
        })

    standings_df = (
        pd.DataFrame(rows)
        .sort_values(by=["Puan", "AV", "AG", "xGD"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    standings_df.insert(0, "Sıra", np.arange(1, len(standings_df) + 1))

    n_m = len(df)
    tot_goals = int((df["home_goals"] + df["away_goals"]).sum())
    tot_ht_goals = int((df["ht_home_goals"].fillna(0) + df["ht_away_goals"].fillna(0)).sum())
    h_wins = int((df["home_goals"] > df["away_goals"]).sum())
    draws = int((df["home_goals"] == df["away_goals"]).sum())
    a_wins = n_m - h_wins - draws
    over25_m = int(((df["home_goals"] + df["away_goals"]) > 2.5).sum())
    btts_m = int(((df["home_goals"] > 0) & (df["away_goals"] > 0)).sum())

    summary = {
        "season": str(season),
        "season_label": format_season_label(season),
        "has_true_xg": has_true_xg,
        "matches_played": n_m,
        "teams_count": len(standings_df),
        "date_min": pd.to_datetime(df["kickoff_utc"].min()).strftime("%Y-%m-%d"),
        "date_max": pd.to_datetime(df["kickoff_utc"].max()).strftime("%Y-%m-%d"),
        "total_goals": tot_goals,
        "goals_per_match": tot_goals / n_m,
        "ht_goals_per_match": tot_ht_goals / n_m,
        "xg_per_match": float((df["hxg"].fillna(0) + df["axg"].fillna(0)).mean()) if has_true_xg else float(standings_df["xG"].sum() / n_m),
        "home_win_pct": h_wins / n_m * 100.0,
        "draw_pct": draws / n_m * 100.0,
        "away_win_pct": a_wins / n_m * 100.0,
        "over25_pct": over25_m / n_m * 100.0,
        "btts_pct": btts_m / n_m * 100.0,
        "corners_per_match": float((df["hc"].fillna(0) + df["ac"].fillna(0)).mean()),
        "shots_per_match": float((df["hs"].fillna(0) + df["as_"].fillna(0)).mean()),
        "sot_per_match": float((df["hst"].fillna(0) + df["ast"].fillna(0)).mean()),
        "fouls_per_match": float((df["hf"].fillna(0) + df["af"].fillna(0)).mean()),
        "yellow_per_match": float((df["hy"].fillna(0) + df["ay"].fillna(0)).mean()),
        "red_per_match": float((df["hr"].fillna(0) + df["ar"].fillna(0)).mean()),
        "standings": standings_df,
        "matches": pd.DataFrame(match_display_rows),
    }
    return summary
