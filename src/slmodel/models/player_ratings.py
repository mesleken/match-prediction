"""Layer B (Bayesian Player Ratings) & Layer C (Tactical Unit Cohesion / Synergy) simulation engine."""

from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd
from slmodel.config import get_model_config
from slmodel.simulate.score_matrix import score_matrix, posterior_predictive_score_matrix
from slmodel.simulate.markets import one_x_two, over_under, both_teams_to_score, most_likely_scores


class PlayerRatingEngine:
    """
    Manages player ratings, lineup deviations, tactical tandem cohesion,
    and what-if match simulation shifts. Strictly orthogonal (zero shift on benchmark XI).
    """

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        beta_att: float = 0.85,
        beta_def: float = 0.85,
        kappa_cohesion: float = 0.18,
    ) -> None:
        self.db_path = db_path
        self.beta_att = beta_att
        self.beta_def = beta_def
        self.kappa_cohesion = kappa_cohesion
        self._cache_rosters: dict[int, list[dict[str, Any]]] = {}
        self._cache_synergies: dict[int, list[dict[str, Any]]] = {}
        self.load_metadata()

    def load_metadata(self) -> None:
        """Load player rosters and synergy pairs from DuckDB."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            # Check if players table exists
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "players" not in tables:
                return

            p_df = con.execute("""
                SELECT player_id, team_id, name, position, nationality, age,
                       att_rating, def_rating, overall_rating, market_value_eur, is_starter
                FROM players ORDER BY team_id, is_starter DESC, overall_rating DESC
            """).fetchdf()

            s_df = con.execute("""
                SELECT team_id, unit_type, player1_id, player2_id, synergy_score
                FROM team_synergy_units
            """).fetchdf()
        finally:
            con.close()

        for tid, group in p_df.groupby("team_id"):
            self._cache_rosters[int(tid)] = group.to_dict(orient="records")

        if not s_df.empty:
            for tid, group in s_df.groupby("team_id"):
                self._cache_synergies[int(tid)] = group.to_dict(orient="records")

    def get_team_roster(self, team_id: int) -> list[dict[str, Any]]:
        """Return all squad players for a team."""
        if team_id not in self._cache_rosters:
            self.load_metadata()
        return self._cache_rosters.get(team_id, [])

    POS_ORDER: dict[str, int] = {
        "GK": 0, "RB": 1, "CB": 2, "LB": 3,
        "DM": 4, "CM": 5, "AM": 6,
        "RW": 7, "LW": 8, "CF": 9,
    }

    def get_default_lineup(self, team_id: int) -> list[dict[str, Any]]:
        """Return the benchmark 11 starters for a team in tactical formation order (GK -> DEF -> MID -> FWD)."""
        roster = self.get_team_roster(team_id)
        starters = [p for p in roster if p.get("is_starter")]
        if len(starters) < 11 and len(roster) >= 11:
            # Fall back to top 11 by overall rating
            sorted_r = sorted(roster, key=lambda x: x["overall_rating"], reverse=True)
            starters = sorted_r[:11]
        else:
            starters = starters[:11]
        return sorted(
            starters,
            key=lambda p: (self.POS_ORDER.get(p.get("position", "CM"), 99), -float(p.get("overall_rating", 70.0))),
        )

    def get_candidate_substitutes(
        self,
        team_id: int,
        starter_player_id: int,
    ) -> list[dict[str, Any]]:
        """
        Return candidate substitutes from the bench for an injured or benched starter,
        ranked by positional suitability and overall rating, with marginal quality deltas.
        """
        roster = self.get_team_roster(team_id)
        benchmark_starters = self.get_default_lineup(team_id)
        starter = next((p for p in benchmark_starters if p["player_id"] == starter_player_id), None)
        if not starter:
            starter = next((p for p in roster if p["player_id"] == starter_player_id), None)

        starter_pos = starter["position"] if starter else "CM"
        starter_ovr = starter["overall_rating"] if starter else 75.0
        starter_att = starter["att_rating"] if starter else 0.20
        starter_def = starter["def_rating"] if starter else 0.20

        # Bench players = players not in benchmark_starters
        starter_ids = set(p["player_id"] for p in benchmark_starters)
        bench = [p for p in roster if p["player_id"] not in starter_ids]

        # Positional compatibility mapping
        pos_compat = {
            "GK": ["GK"],
            "CB": ["CB", "LB", "RB", "DM"],
            "LB": ["LB", "RB", "CB", "LW"],
            "RB": ["RB", "LB", "CB", "RW"],
            "DM": ["DM", "CM", "CB"],
            "CM": ["CM", "AM", "DM"],
            "AM": ["AM", "CM", "RW", "LW"],
            "CF": ["CF", "RW", "LW", "AM"],
            "RW": ["RW", "LW", "CF", "AM"],
            "LW": ["LW", "RW", "CF", "AM"],
        }
        compat_list = pos_compat.get(starter_pos, [starter_pos])

        def sort_key(p):
            pos = p["position"]
            if pos == starter_pos:
                tier = 0
            elif pos in compat_list:
                tier = 1
            else:
                tier = 2
            return (tier, -p["overall_rating"], -p["market_value_eur"])

        sorted_bench = sorted(bench, key=sort_key)
        candidates = []
        for p in sorted_bench:
            cand = dict(p)
            cand["delta_ovr"] = round(float(p["overall_rating"] - starter_ovr), 1)
            cand["delta_att"] = round(float(p["att_rating"] - starter_att), 2)
            cand["delta_def"] = round(float(p["def_rating"] - starter_def), 2)
            candidates.append(cand)

        return candidates

    def compute_lineup_adjustment(
        self,
        team_id: int,
        active_player_ids: list[int],
    ) -> dict[str, Any]:
        """
        Compute orthogonal offensive and defensive shifts wrt the team's benchmark starting 11.
        Delta = 0 when active lineup matches benchmark lineup exactly.
        """
        roster = self.get_team_roster(team_id)
        if not roster:
            return {
                "delta_att": 0.0,
                "delta_def": 0.0,
                "cohesion_score": 1.0,
                "missing_starters": [],
                "substitutions": [],
            }

        roster_map = {p["player_id"]: p for p in roster}
        benchmark_starters = self.get_default_lineup(team_id)
        benchmark_ids = set(p["player_id"] for p in benchmark_starters)
        active_id_set = set(active_player_ids)

        # Baseline sums for benchmark starters
        base_att_sum = sum(p["att_rating"] for p in benchmark_starters)
        base_def_sum = sum(p["def_rating"] for p in benchmark_starters)

        # Active player sums
        active_players = [roster_map[pid] for pid in active_player_ids if pid in roster_map]
        active_att_sum = sum(p["att_rating"] for p in active_players)
        active_def_sum = sum(p["def_rating"] for p in active_players)

        # Pure rating deltas (Layer B)
        diff_att = active_att_sum - base_att_sum
        diff_def = active_def_sum - base_def_sum
        delta_b_att = self.beta_att * diff_att
        delta_b_def = self.beta_def * diff_def

        # Unit synergy and cohesion (Layer C)
        synergies = self._cache_synergies.get(team_id, [])
        bench_syn_sum = 0.0
        active_syn_sum = 0.0

        for s in synergies:
            p1, p2, val = s["player1_id"], s["player2_id"], s["synergy_score"]
            syn_bonus = max(0.0, val - 1.0)
            if p1 in benchmark_ids and p2 in benchmark_ids:
                bench_syn_sum += syn_bonus
            if p1 in active_id_set and p2 in active_id_set:
                active_syn_sum += syn_bonus

        delta_c = self.kappa_cohesion * (active_syn_sum - bench_syn_sum)
        # Defense cohesion protects against goals; attack cohesion creates chances
        delta_c_att = delta_c * 0.50
        delta_c_def = delta_c * 0.50

        # Missing starters analysis
        missing_starters = [p for p in benchmark_starters if p["player_id"] not in active_id_set]
        new_insertions = [p for p in active_players if p["player_id"] not in benchmark_ids]

        net_att_shift = delta_b_att + delta_c_att
        net_def_shift = delta_b_def + delta_c_def

        return {
            "delta_att": float(net_att_shift),
            "delta_def": float(net_def_shift),
            "delta_b_att": float(delta_b_att),
            "delta_b_def": float(delta_b_def),
            "delta_c_att": float(delta_c_att),
            "delta_c_def": float(delta_c_def),
            "active_count": len(active_players),
            "missing_starters": missing_starters,
            "substitutions": new_insertions,
        }

    def compute_availability_lineup_adjustment(
        self,
        team_id: int,
        match_id: Optional[int] = None,
        as_of: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Compute expected attack and defense shifts from player availability
        (manual overrides, injuries, suspensions) compared to the benchmark starting 11.
        Applies Bayesian expected value blending for doubtful players and enforces hard bounds.
        """
        from slmodel.models.availability import AvailabilityResolver
        cfg = get_model_config()

        resolver = AvailabilityResolver(db_path=self.db_path)
        roster_avail = resolver.resolve_team_availability(team_id, match_id=match_id, as_of=as_of)
        avail_map = {p["player_id"]: p for p in roster_avail}

        benchmark_starters = self.get_default_lineup(team_id)
        candidate_bench = [p for p in roster_avail if p["player_id"] not in set(s["player_id"] for s in benchmark_starters)]

        delta_att_total = 0.0
        delta_def_total = 0.0
        absent_starters = []

        max_single = cfg.availability.max_single_player_lambda_shift
        max_team = cfg.availability.max_team_lambda_shift

        for starter in benchmark_starters:
            pid = starter["player_id"]
            p_rec = avail_map.get(pid, starter)
            p_play = float(p_rec.get("p_play", 1.0))

            if p_play < 0.99:
                absent_starters.append({
                    "player_id": pid,
                    "name": starter["name"],
                    "position": starter["position"],
                    "status": p_rec.get("status", "out"),
                    "p_play": p_play,
                    "reason": p_rec.get("reason", ""),
                })

                # Quality drop vs candidate substitute
                subs = self.get_candidate_substitutes(team_id, pid)
                best_sub = subs[0] if subs else None
                sub_att = best_sub["att_rating"] if best_sub else (starter["att_rating"] * 0.70)
                sub_def = best_sub["def_rating"] if best_sub else (starter["def_rating"] * 0.70)

                # Expected delta: (1 - p_play) * (sub_rating - starter_rating)
                prob_absent = 1.0 - p_play
                d_att = prob_absent * (sub_att - starter["att_rating"])
                d_def = prob_absent * (sub_def - starter["def_rating"])

                # Enforce single player bound
                d_att = float(np.clip(d_att, -max_single, max_single))
                d_def = float(np.clip(d_def, -max_single, max_single))

                delta_att_total += d_att
                delta_def_total += d_def

        # Enforce team aggregate bound
        delta_att_bounded = float(np.clip(delta_att_total, -max_team, max_team))
        delta_def_bounded = float(np.clip(delta_def_total, -max_team, max_team))

        return {
            "team_id": team_id,
            "delta_att": delta_att_bounded,
            "delta_def": delta_def_bounded,
            "absent_starters_count": len(absent_starters),
            "absent_starters": absent_starters,
            "is_shifted": bool(len(absent_starters) > 0),
        }

    def simulate_match_with_lineups(
        self,
        home_team_id: int,
        away_team_id: int,
        home_player_ids: Optional[list[int]] = None,
        away_player_ids: Optional[list[int]] = None,
        base_lam_h: float = 1.65,
        base_lam_a: float = 1.15,
        rho: float = -0.05,
    ) -> dict[str, Any]:
        """
        Simulate match with custom or benchmark lineups.
        Computes adjusted lambda and mu, joint score matrix, and exact differences vs baseline.
        """
        # Default to benchmark starting 11 if not provided
        if home_player_ids is None:
            home_starters = self.get_default_lineup(home_team_id)
            home_player_ids = [p["player_id"] for p in home_starters]
        if away_player_ids is None:
            away_starters = self.get_default_lineup(away_team_id)
            away_player_ids = [p["player_id"] for p in away_starters]

        adj_h = self.compute_lineup_adjustment(home_team_id, home_player_ids)
        adj_a = self.compute_lineup_adjustment(away_team_id, away_player_ids)

        # Baseline predictions (Layer A + E)
        M_base = score_matrix(base_lam_h, base_lam_a, rho=rho)
        p1_b, px_b, p2_b = one_x_two(M_base)
        ou25_b = over_under(M_base, 2.5)

        # Adjusted intensities (Layer A + B + C + E):
        # log lam = log lam_base + delta_att(H) - delta_def(A)
        # log mu  = log mu_base  + delta_att(A) - delta_def(H)
        eta_h = np.log(max(0.1, base_lam_h)) + adj_h["delta_att"] - adj_a["delta_def"]
        eta_a = np.log(max(0.1, base_lam_a)) + adj_a["delta_att"] - adj_h["delta_def"]

        adj_lam_h = float(np.clip(np.exp(eta_h), 0.15, 6.0))
        adj_lam_a = float(np.clip(np.exp(eta_a), 0.15, 6.0))

        # Adjusted score matrix and markets
        M_adj = score_matrix(adj_lam_h, adj_lam_a, rho=rho)
        p1_a, px_a, p2_a = one_x_two(M_adj)
        ou25_a = over_under(M_adj, 2.5)
        btts_a = both_teams_to_score(M_adj)
        top5 = most_likely_scores(M_adj, top_n=5)

        return {
            "base": {
                "lam_h": base_lam_h,
                "lam_a": base_lam_a,
                "p_home": float(p1_b),
                "p_draw": float(px_b),
                "p_away": float(p2_b),
                "over_25": float(ou25_b["over"]),
            },
            "adjusted": {
                "lam_h": adj_lam_h,
                "lam_a": adj_lam_a,
                "p_home": float(p1_a),
                "p_draw": float(px_a),
                "p_away": float(p2_a),
                "over_25": float(ou25_a["over"]),
                "btts": float(btts_a["yes"]),
                "top5_scores": top5,
            },
            "shifts": {
                "delta_lam_h": adj_lam_h - base_lam_h,
                "delta_lam_a": adj_lam_a - base_lam_a,
                "delta_p_home": float(p1_a - p1_b),
                "delta_p_draw": float(px_a - px_b),
                "delta_p_away": float(p2_a - p2_b),
                "delta_over_25": float(ou25_a["over"] - ou25_b["over"]),
            },
            "home_adjustment": adj_h,
            "away_adjustment": adj_a,
            "M_adj": M_adj,
        }

    def evaluate_player_importance(
        self,
        team_id: int,
        base_lam_for: float = 1.70,
        base_lam_against: float = 1.10,
    ) -> list[dict[str, Any]]:
        """
        Rank all starters by their marginal value to the team (goal difference and win% drop if absent).
        """
        roster = self.get_team_roster(team_id)
        starters = self.get_default_lineup(team_id)
        starter_ids = [p["player_id"] for p in starters]

        results = []
        base_sim = self.simulate_match_with_lineups(
            home_team_id=team_id,
            away_team_id=9999,
            home_player_ids=starter_ids,
            away_player_ids=[],
            base_lam_h=base_lam_for,
            base_lam_a=base_lam_against,
        )
        base_p_win = base_sim["base"]["p_home"]
        base_gd = base_lam_for - base_lam_against

        bench_players = [p for p in roster if p["player_id"] not in set(starter_ids)]

        for st in starters:
            pid = st["player_id"]
            # Find best positional backup or best bench player
            same_pos_bench = [p for p in bench_players if p["position"] == st["position"]]
            backup = same_pos_bench[0] if same_pos_bench else (bench_players[0] if bench_players else None)

            hypo_ids = [x for x in starter_ids if x != pid]
            if backup:
                hypo_ids.append(backup["player_id"])

            hypo_sim = self.simulate_match_with_lineups(
                home_team_id=team_id,
                away_team_id=9999,
                home_player_ids=hypo_ids,
                away_player_ids=[],
                base_lam_h=base_lam_for,
                base_lam_a=base_lam_against,
            )

            p_win_without = hypo_sim["adjusted"]["p_home"]
            gd_without = hypo_sim["adjusted"]["lam_h"] - hypo_sim["adjusted"]["lam_a"]

            drop_win_pct = (base_p_win - p_win_without) * 100.0
            drop_gd = base_gd - gd_without

            results.append({
                "player_id": pid,
                "name": st["name"],
                "position": st["position"],
                "overall": st["overall_rating"],
                "att_rating": st["att_rating"],
                "def_rating": st["def_rating"],
                "backup_name": backup["name"] if backup else "None",
                "marginal_drop_win_pct": float(drop_win_pct),
                "marginal_goal_diff_loss": float(drop_gd),
            })

        return sorted(results, key=lambda x: x["marginal_goal_diff_loss"], reverse=True)
