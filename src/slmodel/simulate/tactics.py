"""
Tactical profiles and dynamic in-game adaptations conditioned on game state.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TacticalProfile:
    """Tactical setup influencing event probabilities and transition rates."""
    tempo: float = 1.0               # Game pace (affects events per minute)
    line_height: float = 0.5         # 0.0: deep low block, 1.0: aggressive high line
    pressing_intensity: float = 0.5  # 0.0: passive standoff, 1.0: gegenpressing
    risk_appetite: float = 0.5       # 0.0: safe possession, 1.0: direct vertical risk


def adjust_tactics_for_game_state(
    base: TacticalProfile,
    score_diff: int,
    minute: int,
    own_red_cards: int = 0,
    opp_red_cards: int = 0,
) -> TacticalProfile:
    """
    Adjust team tactical profile dynamically according to game context.
    - Trailing teams push numbers forward and take progressive risks as time elapses.
    - Leading teams drop line height to protect leads in final 20 minutes.
    - Red cards force tactical consolidation and shape containment.
    """
    adj_tempo = base.tempo
    adj_line = base.line_height
    adj_press = base.pressing_intensity
    adj_risk = base.risk_appetite

    time_urgency = max(0.0, (minute - 55.0) / 35.0)  # 0.0 at min 55, 1.0 at min 90

    # 1. Trailing scenario (score_diff < 0)
    if score_diff < 0:
        deficit = min(abs(score_diff), 3)
        adj_risk += 0.15 * deficit * time_urgency
        adj_line += 0.12 * deficit * time_urgency
        adj_press += 0.10 * deficit * time_urgency
        adj_tempo += 0.08 * deficit * time_urgency

    # 2. Leading scenario (score_diff > 0)
    elif score_diff > 0:
        lead = min(score_diff, 2)
        adj_risk -= 0.12 * lead * time_urgency
        adj_line -= 0.15 * lead * time_urgency
        adj_tempo -= 0.06 * lead * time_urgency

    # 3. Disciplinary disadvantage (own red cards)
    if own_red_cards > 0:
        adj_line -= 0.20 * own_red_cards
        adj_press -= 0.25 * own_red_cards
        adj_risk -= 0.15 * own_red_cards

    # 4. Numerical advantage (opponent red cards)
    if opp_red_cards > 0:
        adj_line += 0.15 * opp_red_cards
        adj_press += 0.15 * opp_red_cards

    return TacticalProfile(
        tempo=float(np.clip(adj_tempo, 0.6, 1.6)),
        line_height=float(np.clip(adj_line, 0.1, 0.95)),
        pressing_intensity=float(np.clip(adj_press, 0.1, 0.95)),
        risk_appetite=float(np.clip(adj_risk, 0.1, 0.95)),
    )
