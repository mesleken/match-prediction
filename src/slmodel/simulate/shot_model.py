"""
Probabilistic shot generation and expected goals (xG) resolution model.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from slmodel.simulate.state import PitchZone


@dataclass
class ShotResolution:
    is_goal: bool
    is_on_target: bool
    is_saved: bool
    is_blocked: bool
    xg: float


def resolve_shot_attempt(
    zone: PitchZone,
    shooter_att: float = 0.25,
    gk_def: float = 0.25,
    pressure: float = 0.5,
    is_set_piece: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> ShotResolution:
    """
    Resolve a shot attempt into outcome (goal, saved, blocked, off-target) and assign calibrated xG.
    """
    if rng is None:
        rng = np.random.RandomState()

    # 1. Base spatial xG by pitch zone
    if zone == PitchZone.PENALTY_BOX:
        base_xg = float(rng.uniform(0.08, 0.22))
        p_blocked = 0.20 * pressure
        p_target = 0.58 + (shooter_att - 0.20) * 0.3 - pressure * 0.12
    elif zone == PitchZone.ATTACKING_THIRD:
        base_xg = float(rng.uniform(0.025, 0.065))
        p_blocked = 0.28 * pressure
        p_target = 0.38 + (shooter_att - 0.20) * 0.2 - pressure * 0.10
    else:  # Distance speculative shot
        base_xg = float(rng.uniform(0.01, 0.03))
        p_blocked = 0.35
        p_target = 0.25

    if is_set_piece:
        base_xg *= 1.25

    # Striker quality adjustment on xG
    eff_xg = float(np.clip(base_xg * (1.0 + (shooter_att - 0.20) * 0.8), 0.01, 0.95))
    p_target = float(np.clip(p_target, 0.15, 0.85))

    # 2. Simulate shot flight
    roll_block = rng.rand()
    if roll_block < p_blocked:
        return ShotResolution(is_goal=False, is_on_target=False, is_saved=False, is_blocked=True, xg=eff_xg)

    roll_target = rng.rand()
    if roll_target > p_target:
        # Off target
        return ShotResolution(is_goal=False, is_on_target=False, is_saved=False, is_blocked=False, xg=eff_xg)

    # 3. Shot is On-Target: Goalkeeper challenge
    # Base conversion on target is proportional to xG / p_target
    raw_conv_prob = min(eff_xg / max(p_target, 0.1), 0.85)
    # Goalkeeper shot-stopping rating dampens conversion
    gk_factor = 1.0 - (gk_def - 0.20) * 0.7
    final_goal_prob = float(np.clip(raw_conv_prob * gk_factor, 0.05, 0.95))

    roll_goal = rng.rand()
    if roll_goal < final_goal_prob:
        return ShotResolution(is_goal=True, is_on_target=True, is_saved=False, is_blocked=False, xg=eff_xg)
    else:
        return ShotResolution(is_goal=False, is_on_target=True, is_saved=True, is_blocked=False, xg=eff_xg)
