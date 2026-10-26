"""
Calibrated parameters for MatchEngine tactical event flow and shot resolution.
Values calibrated against Süper Lig 3,129-match warehouse moments.
"""

from dataclasses import dataclass


@dataclass
class EngineParams:
    """Calibrated parameters governing pitch-zone semi-Markov transitions and events."""

    # Time per micro-possession (seconds)
    min_sequence_sec: int = 18
    max_sequence_sec: int = 36

    # Defensive Third transitions
    p_def_turnover_base: float = 0.08  # Opponent high-press turnover in attacking third
    p_def_pass_middle: float = 0.78    # Build-up pass to middle third

    # Middle Third transitions
    p_mid_advance_base: float = 0.38   # Progressive entry into attacking third
    p_mid_turnover_base: float = 0.22  # Turnover in middle third
    p_mid_foul_base: float = 0.16      # Midfield tactical/physical foul (~26 fouls/match total)

    # Attacking Third transitions
    p_att_box_base: float = 0.30       # Entry into penalty box (cross/key pass)
    p_att_shot_base: float = 0.24      # Long-range shot outside box
    p_att_corner_base: float = 0.11    # Corner won (deflection/blocked cross)
    p_att_foul_base: float = 0.10      # Attacking third foul
    p_att_recycle_base: float = 0.15   # Recycle back to midfield

    # Penalty Box transitions
    p_box_shot_base: float = 0.56      # Direct shot attempt inside penalty box
    p_box_corner_base: float = 0.16    # Corner awarded (clearance/save parry)
    p_box_foul_base: float = 0.08      # Penalty/foul inside box

    # Shot & Save Outcomes
    box_xg_min: float = 0.09
    box_xg_max: float = 0.25
    att_xg_min: float = 0.03
    att_xg_max: float = 0.075
    save_corner_rebound_prob: float = 0.22  # Probability saved shot is deflected out for corner

    # Discipline & Cards (calibrated to ~4.54 yellow cards and ~0.20 red cards per match)
    p_foul_yellow: float = 0.175       # Yellow card probability given a foul
    p_foul_red: float = 0.008          # Direct red card probability given a foul

    # League reference averages
    league_mean_lambda: float = 1.55   # Historical Süper Lig home goals
    league_mean_mu: float = 1.275      # Historical Süper Lig away goals


DEFAULT_ENGINE_PARAMS = EngineParams()
