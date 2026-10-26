"""
Monte Carlo dynamic multi-scenario match simulation runner.
Executes thousands of event-driven matches to extract empirical joint distributions,
goal timing curves, game-state comeback rates, and side-market expectations.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from slmodel.simulate.match_engine import MatchEngine, MatchTimeline
from slmodel.simulate.tactics import TacticalProfile
from slmodel.simulate.params import EngineParams, DEFAULT_ENGINE_PARAMS


@dataclass
class MonteCarloResult:
    n_simulations: int
    p_home: float
    p_draw: float
    p_away: float
    mean_goals_h: float
    mean_goals_a: float
    mean_xg_h: float
    mean_xg_a: float
    score_matrix: np.ndarray  # 11x11 joint distribution
    top_scores: list[dict[str, Any]]
    p_over_15: float
    p_over_25: float
    p_over_35: float
    p_btts_yes: float
    p_btts_no: float
    mean_corners_tot: float
    mean_cards_tot: float
    comeback_pct: float
    goal_timing_h: dict[str, float]
    goal_timing_a: dict[str, float]
    sample_match_timeline: Optional[MatchTimeline] = None


def run_monte_carlo_simulation(
    home_team_id: int,
    away_team_id: int,
    att_h: float = 0.20,
    def_h: float = 0.20,
    att_a: float = 0.20,
    def_a: float = 0.20,
    lam_h: Optional[float] = None,
    lam_a: Optional[float] = None,
    tactics_h: Optional[TacticalProfile] = None,
    tactics_a: Optional[TacticalProfile] = None,
    params: Optional[EngineParams] = None,
    n_simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run N event-based match simulations and aggregate scenario distributions.
    """
    rng = np.random.RandomState(seed)
    engine = MatchEngine(rng=rng, params=params)

    matrix = np.zeros((11, 11), dtype=np.float64)
    outcomes = [0, 0, 0]  # [H, D, A]
    goals_h_list = []
    goals_a_list = []
    xg_h_list = []
    xg_a_list = []
    corners_list = []
    cards_list = []
    comebacks = 0

    timing_bins = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
    timing_counts_h = {b: 0 for b in timing_bins}
    timing_counts_a = {b: 0 for b in timing_bins}

    def _bin_minute(minute: int) -> str:
        if minute <= 15: return "0-15"
        elif minute <= 30: return "15-30"
        elif minute <= 45: return "30-45"
        elif minute <= 60: return "45-60"
        elif minute <= 75: return "60-75"
        return "75-90+"

    sample_timeline: Optional[MatchTimeline] = None

    for i in range(n_simulations):
        timeline = engine.simulate_match(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            att_h=att_h,
            def_h=def_h,
            att_a=att_a,
            def_a=def_a,
            lam_h=lam_h,
            lam_a=lam_a,
            tactics_h=tactics_h,
            tactics_a=tactics_a,
        )
        if i == 0:
            sample_timeline = timeline

        gh = min(timeline.score_h, 10)
        ga = min(timeline.score_a, 10)
        matrix[gh, ga] += 1.0

        if gh > ga: outcomes[0] += 1
        elif gh == ga: outcomes[1] += 1
        else: outcomes[2] += 1

        goals_h_list.append(timeline.score_h)
        goals_a_list.append(timeline.score_a)
        xg_h_list.append(timeline.stats.xg_h)
        xg_a_list.append(timeline.stats.xg_a)
        corners_list.append(timeline.stats.corners_h + timeline.stats.corners_a)
        cards_list.append(timeline.stats.yellow_cards_h + timeline.stats.yellow_cards_a + timeline.stats.red_cards_h + timeline.stats.red_cards_a)

        # Check comeback: Trailed at half-time but didn't lose
        if timeline.ht_score_h < timeline.ht_score_a and timeline.score_h >= timeline.score_a:
            comebacks += 1
        elif timeline.ht_score_a < timeline.ht_score_h and timeline.score_a >= timeline.score_h:
            comebacks += 1

        for m in timeline.stats.goal_minutes_h:
            timing_counts_h[_bin_minute(m)] += 1
        for m in timeline.stats.goal_minutes_a:
            timing_counts_a[_bin_minute(m)] += 1

    matrix /= float(n_simulations)
    p_h = outcomes[0] / n_simulations
    p_d = outcomes[1] / n_simulations
    p_a = outcomes[2] / n_simulations

    # Top scores
    score_tuples = []
    for h in range(11):
        for a in range(11):
            if matrix[h, a] > 0:
                score_tuples.append((h, a, float(matrix[h, a])))
    score_tuples.sort(key=lambda x: x[2], reverse=True)
    top_scores = [{"home_goals": h, "away_goals": a, "score": f"{h}-{a}", "probability": p} for h, a, p in score_tuples[:5]]

    tot_goals = np.array(goals_h_list) + np.array(goals_a_list)
    p_over_15 = float(np.mean(tot_goals > 1.5))
    p_over_25 = float(np.mean(tot_goals > 2.5))
    p_over_35 = float(np.mean(tot_goals > 3.5))

    btts_arr = (np.array(goals_h_list) > 0) & (np.array(goals_a_list) > 0)
    p_btts_yes = float(np.mean(btts_arr))
    p_btts_no = 1.0 - p_btts_yes

    tot_h_goals = max(sum(timing_counts_h.values()), 1)
    tot_a_goals = max(sum(timing_counts_a.values()), 1)

    timing_dist_h = {k: round(v / tot_h_goals, 3) for k, v in timing_counts_h.items()}
    timing_dist_a = {k: round(v / tot_a_goals, 3) for k, v in timing_counts_a.items()}

    return MonteCarloResult(
        n_simulations=n_simulations,
        p_home=round(p_h, 3),
        p_draw=round(p_d, 3),
        p_away=round(p_a, 3),
        mean_goals_h=round(float(np.mean(goals_h_list)), 2),
        mean_goals_a=round(float(np.mean(goals_a_list)), 2),
        mean_xg_h=round(float(np.mean(xg_h_list)), 2),
        mean_xg_a=round(float(np.mean(xg_a_list)), 2),
        score_matrix=matrix,
        top_scores=top_scores,
        p_over_15=round(p_over_15, 3),
        p_over_25=round(p_over_25, 3),
        p_over_35=round(p_over_35, 3),
        p_btts_yes=round(p_btts_yes, 3),
        p_btts_no=round(p_btts_no, 3),
        mean_corners_tot=round(float(np.mean(corners_list)), 1),
        mean_cards_tot=round(float(np.mean(cards_list)), 1),
        comeback_pct=round(float(comebacks / n_simulations * 100.0), 1),
        goal_timing_h=timing_dist_h,
        goal_timing_a=timing_dist_a,
        sample_match_timeline=sample_timeline,
    )
