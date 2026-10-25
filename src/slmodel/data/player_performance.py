"""Data-driven player match performance tracking, statistical aggregation, and Bayesian rating updates for 2026/27 Süper Lig."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd


def generate_and_sync_player_match_stats(
    db_path: str = "data/warehouse.duckdb",
    season: str = "2627",
) -> dict[str, Any]:
    """
    Generate realistic, reproducible player-level match statistics for all played matches
    in the given season that strictly match team scores, shots, shots on target, and cards.
    Populates 'player_match_stats' table in DuckDB.
    """
    con = duckdb.connect(db_path)
    try:
        # Load all matches of the season with team match stats
        q_matches = """
            SELECT
                m.match_id, m.home_team_id, m.away_team_id,
                m.home_goals, m.away_goals,
                COALESCE(sh.shots, 12) AS hs, COALESCE(sh.shots_on_target, 4) AS hst,
                COALESCE(sh.corners, 5) AS hc, COALESCE(sh.fouls, 12) AS hf,
                COALESCE(sh.yellow, 2) AS hy, COALESCE(sh.red, 0) AS hr,
                COALESCE(sh.xg, 1.25) AS hxg,
                COALESCE(sa.shots, 10) AS as_, COALESCE(sa.shots_on_target, 3) AS ast,
                COALESCE(sa.corners, 4) AS ac, COALESCE(sa.fouls, 13) AS af,
                COALESCE(sa.yellow, 2) AS ay, COALESCE(sa.red, 0) AS ar,
                COALESCE(sa.xg, 1.05) AS axg
            FROM matches m
            LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
            LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
            WHERE m.season = ? AND m.status = 'played'
              AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
            ORDER BY m.match_id ASC
        """
        matches_df = con.execute(q_matches, [str(season)]).fetchdf()

        # Load all players
        p_df = con.execute("""
            SELECT player_id, team_id, name, position, att_rating, def_rating, overall_rating, is_starter
            FROM players ORDER BY team_id, is_starter DESC, overall_rating DESC
        """).fetchdf()
    finally:
        con.close()

    if matches_df.empty or p_df.empty:
        return {"matches_processed": 0, "records_generated": 0}

    # Group players by team
    team_rosters: dict[int, list[dict[str, Any]]] = {}
    for tid, group in p_df.groupby("team_id"):
        team_rosters[int(tid)] = group.to_dict(orient="records")

    match_stats_rows: list[dict[str, Any]] = []
    stat_id_counter = 0

    for _, m in matches_df.iterrows():
        mid = int(m["match_id"])
        h_id = int(m["home_team_id"])
        a_id = int(m["away_team_id"])
        h_goals = int(m["home_goals"])
        a_goals = int(m["away_goals"])

        h_shots = max(h_goals, int(m["hs"]))
        h_sot = max(h_goals, min(h_shots, int(m["hst"])))
        h_yellow = int(m["hy"])
        h_red = int(m["hr"])
        h_xg = max(0.05, float(m["hxg"]))

        a_shots = max(a_goals, int(m["as_"]))
        a_sot = max(a_goals, min(a_shots, int(m["ast"])))
        a_yellow = int(m["ay"])
        a_red = int(m["ar"])
        a_xg = max(0.05, float(m["axg"]))

        # Deterministic RNG seeded by match_id for reproducibility
        rng = np.random.RandomState(mid * 997 + 101)

        # Process Home Team
        h_players = _simulate_team_match_events(
            rng=rng,
            match_id=mid,
            team_id=h_id,
            roster=team_rosters.get(h_id, []),
            goals_scored=h_goals,
            goals_conceded=a_goals,
            team_shots=h_shots,
            team_sot=h_sot,
            opp_sot=a_sot,
            yellow_cards=h_yellow,
            red_cards=h_red,
            team_xg=h_xg,
        )

        # Process Away Team
        a_players = _simulate_team_match_events(
            rng=rng,
            match_id=mid,
            team_id=a_id,
            roster=team_rosters.get(a_id, []),
            goals_scored=a_goals,
            goals_conceded=h_goals,
            team_shots=a_shots,
            team_sot=a_sot,
            opp_sot=h_sot,
            yellow_cards=a_yellow,
            red_cards=a_red,
            team_xg=a_xg,
        )

        for p_stat in h_players + a_players:
            stat_id_counter += 1
            p_stat["stat_id"] = stat_id_counter
            match_stats_rows.append(p_stat)

    # Bulk insert into DuckDB
    con = duckdb.connect(db_path)
    try:
        # Clear existing stats for these matches
        m_ids_str = ",".join(str(int(mid)) for mid in matches_df["match_id"])
        con.execute(f"DELETE FROM player_match_stats WHERE match_id IN ({m_ids_str})")

        df_insert = pd.DataFrame(match_stats_rows)
        con.register("tmp_p_stats", df_insert)
        con.execute("""
            INSERT INTO player_match_stats
            SELECT
                stat_id, match_id, player_id, team_id, is_starter, minutes_played,
                goals, assists, shots, shots_on_target, key_passes,
                tackles, interceptions, clearances, saves, clean_sheet, goals_conceded,
                fouls_committed, yellow_cards, red_cards, xg, xa, match_rating
            FROM tmp_p_stats
        """)
        con.unregister("tmp_p_stats")
        con.commit()
    finally:
        con.close()

    return {
        "matches_processed": len(matches_df),
        "records_generated": len(match_stats_rows),
    }


def _simulate_team_match_events(
    rng: np.random.RandomState,
    match_id: int,
    team_id: int,
    roster: list[dict[str, Any]],
    goals_scored: int,
    goals_conceded: int,
    team_shots: int,
    team_sot: int,
    opp_sot: int,
    yellow_cards: int,
    red_cards: int,
    team_xg: float,
) -> list[dict[str, Any]]:
    """Simulate individual events for a team in a match matching team totals."""
    if not roster:
        return []

    # Select 11 starters: prefer is_starter=True, else highest overall_rating
    starters = [p for p in roster if p.get("is_starter")]
    bench = [p for p in roster if not p.get("is_starter")]

    if len(starters) < 11:
        needed = 11 - len(starters)
        starters.extend(bench[:needed])
        bench = bench[needed:]

    # Ensure 1 GK
    gks = [p for p in starters if p["position"] == "GK"]
    if not gks:
        bench_gks = [p for p in bench if p["position"] == "GK"]
        if bench_gks:
            starters[0] = bench_gks[0]
            bench.remove(bench_gks[0])

    active_starters = starters[:11]

    # Select 3 to 4 substitutes who get playing time (10-30 mins)
    n_subs = min(len(bench), rng.randint(3, 5))
    active_subs = list(rng.choice(bench, size=n_subs, replace=False)) if n_subs > 0 else []

    playing_players: list[dict[str, Any]] = []

    # Starters minutes
    for p in active_starters:
        is_gk = p["position"] == "GK"
        mins = 90 if is_gk else rng.choice([90, 90, 85, 78, 72, 68])
        playing_players.append({
            "player": p,
            "is_starter": True,
            "minutes": int(mins),
        })

    # Subs minutes
    for p in active_subs:
        mins = rng.choice([28, 22, 18, 14, 10])
        playing_players.append({
            "player": p,
            "is_starter": False,
            "minutes": int(mins),
        })

    # Initialize stats dict for each player
    player_stats: dict[int, dict[str, Any]] = {}
    for entry in playing_players:
        p = entry["player"]
        pid = p["player_id"]
        pos = p["position"]
        mins = entry["minutes"]
        is_st = entry["is_starter"]

        player_stats[pid] = {
            "match_id": match_id,
            "player_id": pid,
            "team_id": team_id,
            "is_starter": is_st,
            "minutes_played": mins,
            "goals": 0,
            "assists": 0,
            "shots": 0,
            "shots_on_target": 0,
            "key_passes": 0,
            "tackles": 0,
            "interceptions": 0,
            "clearances": 0,
            "saves": 0,
            "clean_sheet": 0,
            "goals_conceded": goals_conceded if (is_st or mins >= 30) else (1 if goals_conceded > 1 else 0),
            "fouls_committed": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "xg": 0.0,
            "xa": 0.0,
            "match_rating": 6.70,
            "_pos": pos,
            "_att": float(p.get("att_rating", 0.1)),
            "_def": float(p.get("def_rating", 0.1)),
            "_ovr": float(p.get("overall_rating", 75.0)),
        }

    # Clean sheet: if 0 conceded, GK and starters get clean_sheet = 1
    if goals_conceded == 0:
        for pid, s in player_stats.items():
            if s["_pos"] in ("GK", "CB", "LB", "RB") and s["minutes_played"] >= 60:
                s["clean_sheet"] = 1

    # Goalkeeper saves: opp_sot - goals_conceded
    gk_saves = max(0, opp_sot - goals_conceded)
    for pid, s in player_stats.items():
        if s["_pos"] == "GK":
            s["saves"] = gk_saves
            break

    # Distribute goals among attackers and midfielders
    if goals_scored > 0:
        goal_weights = []
        candidates = []
        for pid, s in player_stats.items():
            if s["_pos"] == "GK":
                continue
            pos = s["_pos"]
            base_w = 0.50 if pos == "CF" else (0.22 if pos in ("RW", "LW") else (0.15 if pos == "AM" else (0.08 if pos == "CM" else 0.03)))
            w = base_w * (1.0 + s["_att"] * 2.0) * (s["minutes_played"] / 90.0)
            candidates.append(pid)
            goal_weights.append(max(0.01, w))

        gw_arr = np.array(goal_weights) / sum(goal_weights)
        for _ in range(goals_scored):
            scorer_id = rng.choice(candidates, p=gw_arr)
            player_stats[scorer_id]["goals"] += 1
            player_stats[scorer_id]["shots_on_target"] += 1
            player_stats[scorer_id]["shots"] += 1
            player_stats[scorer_id]["xg"] += round(float(rng.uniform(0.35, 0.75)), 2)

        # Distribute assists (about 70% of goals have an assist)
        for _ in range(goals_scored):
            if rng.rand() < 0.70:
                assist_weights = []
                a_cand = []
                for pid, s in player_stats.items():
                    if s["_pos"] == "GK":
                        continue
                    pos = s["_pos"]
                    base_aw = 0.35 if pos in ("AM", "RW", "LW") else (0.25 if pos in ("CM", "LB", "RB") else 0.15)
                    aw = base_aw * (s["minutes_played"] / 90.0)
                    a_cand.append(pid)
                    assist_weights.append(max(0.01, aw))
                aw_arr = np.array(assist_weights) / sum(assist_weights)
                assister_id = rng.choice(a_cand, p=aw_arr)
                player_stats[assister_id]["assists"] += 1
                player_stats[assister_id]["key_passes"] += 1
                player_stats[assister_id]["xa"] += round(float(rng.uniform(0.20, 0.50)), 2)

    # Distribute remaining shots and shots on target
    current_shots = sum(s["shots"] for s in player_stats.values())
    current_sot = sum(s["shots_on_target"] for s in player_stats.values())
    rem_shots = max(0, team_shots - current_shots)
    rem_sot = max(0, team_sot - current_sot)

    field_pids = [pid for pid, s in player_stats.items() if s["_pos"] != "GK"]
    if field_pids:
        # Distribute remaining shots
        for _ in range(rem_shots):
            pid = rng.choice(field_pids)
            player_stats[pid]["shots"] += 1
            player_stats[pid]["xg"] += round(float(rng.uniform(0.04, 0.15)), 2)

        # Distribute remaining shots on target
        for _ in range(rem_sot):
            pid = rng.choice(field_pids)
            player_stats[pid]["shots_on_target"] += 1
            if player_stats[pid]["shots"] < player_stats[pid]["shots_on_target"]:
                player_stats[pid]["shots"] = player_stats[pid]["shots_on_target"]

    # Key passes, tackles, interceptions
    for pid, s in player_stats.items():
        pos = s["_pos"]
        mins = s["minutes_played"]
        if pos == "GK":
            continue
        # Key passes
        kp_lambda = 1.8 if pos in ("AM", "RW", "LW") else (1.2 if pos == "CM" else 0.4)
        s["key_passes"] += int(rng.poisson(kp_lambda * (mins / 90.0)))

        # Tackles & Interceptions
        def_lambda = 3.5 if pos in ("CB", "DM", "LB", "RB") else (1.8 if pos == "CM" else 0.6)
        s["tackles"] += int(rng.poisson(def_lambda * (mins / 90.0) * 0.6))
        s["interceptions"] += int(rng.poisson(def_lambda * (mins / 90.0) * 0.4))
        if pos in ("CB", "LB", "RB"):
            s["clearances"] += int(rng.poisson(3.0 * (mins / 90.0)))

    # Distribute Yellow and Red cards
    if yellow_cards > 0 and field_pids:
        for _ in range(yellow_cards):
            pid = rng.choice(field_pids)
            player_stats[pid]["yellow_cards"] += 1
            player_stats[pid]["fouls_committed"] += rng.randint(1, 3)

    if red_cards > 0 and field_pids:
        for _ in range(red_cards):
            pid = rng.choice(field_pids)
            player_stats[pid]["red_cards"] += 1

    # Compute Sofascore-style Match Rating
    for pid, s in player_stats.items():
        base_r = 6.65
        # Goal bonus
        base_r += s["goals"] * 0.85
        # Assist bonus
        base_r += s["assists"] * 0.50
        # Shot on target
        base_r += s["shots_on_target"] * 0.10
        # Key passes
        base_r += s["key_passes"] * 0.12
        # GK saves
        base_r += s["saves"] * 0.22
        # Clean sheet bonus (GK and DEF)
        if s["clean_sheet"] == 1:
            base_r += 0.50
        # Conceded penalty
        if s["_pos"] in ("GK", "CB", "LB", "RB", "DM"):
            base_r -= s["goals_conceded"] * 0.22
        # Defensive interventions
        base_r += (s["tackles"] + s["interceptions"]) * 0.05
        # Cards
        base_r -= s["yellow_cards"] * 0.40
        base_r -= s["red_cards"] * 1.50

        # Sub penalty if low minutes and no action
        if s["minutes_played"] < 30 and s["goals"] == 0 and s["assists"] == 0:
            base_r = min(base_r, 6.70)

        # Clamping
        s["match_rating"] = round(float(np.clip(base_r, 5.5, 9.6)), 2)
        s["xg"] = round(float(s["xg"]), 2)
        s["xa"] = round(float(s["xa"]), 2)

        # Clean internal keys
        del s["_pos"]
        del s["_att"]
        del s["_def"]
        del s["_ovr"]

    return list(player_stats.values())


def compute_player_season_stats_and_ratings(
    db_path: str = "data/warehouse.duckdb",
    season: str = "2627",
    update_players_table: bool = False,
) -> pd.DataFrame:
    """
    Aggregate all match statistics for the season and compute empirical data-driven ratings
    with Bayesian shrinkage towards position baselines. Updates strictly 'player_season_stats' table.
    
    ARCHITECTURAL GUARANTEE: Never writes to 'players' table or squads JSON to strictly prevent
    circular future data leakage into pre-season squad priors and match predictions.
    `update_players_table` is deprecated and ignored for backwards compatibility.
    """
    con = duckdb.connect(db_path)
    try:
        # Load player metadata
        p_df = con.execute("""
            SELECT player_id, team_id, name, position, nationality, age,
                   att_rating AS prior_att, def_rating AS prior_def, overall_rating AS prior_ovr,
                   market_value_eur, is_starter
            FROM players ORDER BY player_id
        """).fetchdf()

        # Aggregate match statistics from player_match_stats
        agg_df = con.execute("""
            SELECT
                player_id,
                COUNT(match_id) AS matches_played,
                SUM(CASE WHEN is_starter THEN 1 ELSE 0 END) AS matches_started,
                SUM(minutes_played) AS minutes_played,
                SUM(goals) AS goals,
                SUM(assists) AS assists,
                SUM(shots) AS shots,
                SUM(shots_on_target) AS shots_on_target,
                SUM(key_passes) AS key_passes,
                SUM(tackles) AS tackles,
                SUM(interceptions) AS interceptions,
                SUM(saves) AS saves,
                SUM(clean_sheet) AS clean_sheets,
                SUM(goals_conceded) AS goals_conceded,
                SUM(yellow_cards) AS yellow_cards,
                SUM(red_cards) AS red_cards,
                ROUND(SUM(xg), 2) AS xg,
                ROUND(SUM(xa), 2) AS xa,
                ROUND(AVG(match_rating), 2) AS avg_match_rating
            FROM player_match_stats
            GROUP BY player_id
        """).fetchdf()
    finally:
        con.close()

    if p_df.empty:
        return pd.DataFrame()

    # Merge metadata with aggregates (players with 0 matches get 0s)
    df = pd.merge(p_df, agg_df, on="player_id", how="left").fillna({
        "matches_played": 0,
        "matches_started": 0,
        "minutes_played": 0,
        "goals": 0,
        "assists": 0,
        "shots": 0,
        "shots_on_target": 0,
        "key_passes": 0,
        "tackles": 0,
        "interceptions": 0,
        "saves": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "xg": 0.0,
        "xa": 0.0,
        "avg_match_rating": 6.70,
    })

    # Integer conversions
    int_cols = [
        "matches_played", "matches_started", "minutes_played", "goals", "assists",
        "shots", "shots_on_target", "key_passes", "tackles", "interceptions",
        "saves", "clean_sheets", "goals_conceded", "yellow_cards", "red_cards",
    ]
    for c in int_cols:
        df[c] = df[c].astype(int)

    computed_att = []
    computed_def = []
    computed_ovr = []

    for _, row in df.iterrows():
        pos = row["position"]
        mins = float(row["minutes_played"])
        prior_att = float(row["prior_att"])
        prior_def = float(row["prior_def"])
        prior_ovr = float(row["prior_ovr"])

        # Bayesian rate shrinkage parameter (4 matches equivalent prior)
        k_matches = 4.0
        m90 = max(mins, 1.0) / 90.0
        g = float(row["goals"])
        a = float(row["assists"])
        sot = float(row["shots_on_target"])
        kp = float(row["key_passes"])
        xg = float(row["xg"])
        xa = float(row["xa"])
        r_avg = float(row["avg_match_rating"])
        saves = float(row["saves"])
        cs = float(row["clean_sheets"])
        gc = float(row["goals_conceded"])
        mp = max(float(row["matches_played"]), 1.0)
        tack = float(row["tackles"]) + float(row["interceptions"])

        # Stabilized regressed per-90 rates
        reg_g90 = (g + prior_att * 1.5 * k_matches) / (m90 + k_matches)
        reg_a90 = (a + 0.20 * k_matches) / (m90 + k_matches)
        reg_xg90 = (xg + xa + 0.35 * k_matches) / (m90 + k_matches)
        reg_sot90 = (sot + 0.8 * k_matches) / (m90 + k_matches)
        reg_kp90 = (kp + 1.0 * k_matches) / (m90 + k_matches)

        # 1. ATTACK RATING CALCULATION
        if pos == "CF":
            emp_att = 0.16 + 0.26 * min(1.2, reg_g90) + 0.10 * min(0.8, reg_a90) + 0.08 * reg_xg90 + 0.05 * max(0.0, r_avg - 6.6)
            emp_att = min(emp_att, 0.46)
        elif pos in ("RW", "LW"):
            emp_att = 0.14 + 0.18 * min(1.0, reg_g90) + 0.16 * min(0.8, reg_a90) + 0.06 * reg_xg90 + 0.04 * max(0.0, r_avg - 6.6)
            emp_att = min(emp_att, 0.38)
        elif pos in ("AM", "CM"):
            emp_att = 0.12 + 0.14 * min(0.8, reg_g90) + 0.18 * min(0.8, reg_a90) + 0.05 * reg_kp90 + 0.04 * max(0.0, r_avg - 6.6)
            emp_att = min(emp_att, 0.34)
        elif pos in ("LB", "RB"):
            emp_att = 0.06 + 0.08 * min(0.6, reg_a90) + 0.04 * reg_kp90
            emp_att = min(emp_att, 0.20)
        elif pos == "DM":
            emp_att = 0.05 + 0.06 * min(0.6, reg_a90) + 0.03 * reg_g90
            emp_att = min(emp_att, 0.16)
        elif pos == "CB":
            emp_att = 0.03 + 0.04 * min(0.4, reg_g90)
            emp_att = min(emp_att, 0.10)
        else:  # GK
            emp_att = 0.00

        # Bayesian Shrinkage towards prior (M0 = 360 minutes ~ 4 full matches)
        z_att = mins / (mins + 360.0) if mins > 0 else 0.0
        final_att = round(float(z_att * emp_att + (1.0 - z_att) * prior_att), 2)
        final_att = max(0.0, final_att)

        # 2. DEFENSE RATING CALCULATION
        reg_gc90 = (gc + 1.2 * k_matches) / (m90 + k_matches)
        reg_tack90 = (tack + 2.0 * k_matches) / (m90 + k_matches)
        cs_rate = cs / mp

        if pos == "GK":
            save_pct = saves / max(saves + gc, 1.0)
            emp_def = 0.18 + 0.16 * save_pct + 0.14 * cs_rate - 0.05 * reg_gc90 + 0.04 * (r_avg - 6.6)
            emp_def = np.clip(emp_def, 0.10, 0.42)
        elif pos in ("CB", "DM"):
            emp_def = 0.16 + 0.15 * cs_rate + 0.05 * reg_tack90 - 0.05 * reg_gc90 + 0.04 * (r_avg - 6.6)
            emp_def = np.clip(emp_def, 0.10, 0.42)
        elif pos in ("LB", "RB"):
            emp_def = 0.10 + 0.12 * cs_rate + 0.04 * reg_tack90 - 0.05 * reg_gc90 + 0.03 * (r_avg - 6.6)
            emp_def = np.clip(emp_def, 0.06, 0.32)
        elif pos == "CM":
            emp_def = 0.08 + 0.08 * cs_rate + 0.03 * reg_tack90
            emp_def = np.clip(emp_def, 0.04, 0.26)
        else:  # CF, RW, LW, AM
            emp_def = 0.01 + 0.01 * reg_tack90
            emp_def = np.clip(emp_def, 0.00, 0.06)

        z_def = mins / (mins + 360.0) if mins > 0 else 0.0
        final_def = round(float(z_def * emp_def + (1.0 - z_def) * prior_def), 2)
        final_def = max(0.0, final_def)

        # 3. OVERALL RATING CALCULATION
        emp_ovr = 71.0 + (r_avg - 6.70) * 11.5 + (g * 0.45) + (a * 0.30) + (cs * 0.40) - (gc * 0.12)
        z_ovr = mins / (mins + 360.0) if mins > 0 else 0.0
        final_ovr = round(float(np.clip(z_ovr * emp_ovr + (1.0 - z_ovr) * prior_ovr, 68.0, 88.0)), 1)

        computed_att.append(final_att)
        computed_def.append(final_def)
        computed_ovr.append(final_ovr)

    df["computed_att_rating"] = computed_att
    df["computed_def_rating"] = computed_def
    df["computed_overall_rating"] = computed_ovr
    df["season"] = str(season)

    # Insert into player_season_stats
    con = duckdb.connect(db_path)
    try:
        con.execute("DELETE FROM player_season_stats WHERE season = ?", [str(season)])
        df_season_insert = df[[
            "season", "player_id", "team_id", "matches_played", "matches_started", "minutes_played",
            "goals", "assists", "shots", "shots_on_target", "key_passes",
            "tackles", "interceptions", "saves", "clean_sheets", "goals_conceded",
            "yellow_cards", "red_cards", "xg", "xa", "avg_match_rating",
            "computed_att_rating", "computed_def_rating", "computed_overall_rating"
        ]]
        con.register("tmp_season_stats", df_season_insert)
        con.execute("""
            INSERT INTO player_season_stats
            SELECT * FROM tmp_season_stats
        """)
        con.unregister("tmp_season_stats")

        con.commit()
    finally:
        con.close()

    return df


def get_player_season_leaderboards(
    db_path: str = "data/warehouse.duckdb",
    season: str = "2627",
) -> dict[str, pd.DataFrame]:
    """Retrieve statistical leaderboards for the 2026/27 Süper Lig season."""
    con = duckdb.connect(db_path)
    try:
        q = """
            SELECT
                p.name AS "Oyuncu",
                t.name AS "Takım",
                p.position AS "Mevki",
                s.matches_played AS "Maç",
                s.minutes_played AS "Dakika",
                s.goals AS "Gol",
                s.assists AS "Asist",
                ROUND(s.xg + s.xa, 2) AS "xG+xA",
                s.shots_on_target AS "İsabetli Şut",
                s.saves AS "Kurtarış",
                s.clean_sheets AS "Clean Sheet",
                s.tackles + s.interceptions AS "Top Çalma",
                s.avg_match_rating AS "Maç Puanı",
                s.computed_att_rating AS "Att Rating",
                s.computed_def_rating AS "Def Rating",
                s.computed_overall_rating AS "Genel Reyting"
            FROM player_season_stats s
            JOIN players p ON s.player_id = p.player_id
            JOIN teams t ON s.team_id = t.team_id
            WHERE s.season = ?
        """
        df = con.execute(q, [str(season)]).fetchdf()
    finally:
        con.close()

    if df.empty:
        return {
            "top_scorers": pd.DataFrame(),
            "top_assisters": pd.DataFrame(),
            "top_rated": pd.DataFrame(),
            "best_goalkeepers": pd.DataFrame(),
            "all_players": pd.DataFrame(),
        }

    top_scorers = df.sort_values(by=["Gol", "xG+xA", "Dakika"], ascending=[False, False, True]).head(15).reset_index(drop=True)
    top_scorers.index += 1

    top_assisters = df.sort_values(by=["Asist", "xG+xA", "Dakika"], ascending=[False, False, True]).head(15).reset_index(drop=True)
    top_assisters.index += 1

    top_rated = df[df["Dakika"] >= 180].sort_values(by=["Maç Puanı", "Genel Reyting"], ascending=[False, False]).head(15).reset_index(drop=True)
    top_rated.index += 1

    best_gk = df[df["Mevki"] == "GK"].sort_values(by=["Kurtarış", "Clean Sheet", "Maç Puanı"], ascending=[False, False, False]).head(10).reset_index(drop=True)
    best_gk.index += 1

    return {
        "top_scorers": top_scorers,
        "top_assisters": top_assisters,
        "top_rated": top_rated,
        "best_goalkeepers": best_gk,
        "all_players": df,
    }
