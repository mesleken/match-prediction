"""
Match state and event definitions for event-based dynamic match simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PitchZone(str, Enum):
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"
    PENALTY_BOX = "penalty_box"


class EventType(str, Enum):
    POSSESSION_START = "possession_start"
    PASS_COMPLETED = "pass_completed"
    PROGRESSIVE_CARRY = "progressive_carry"
    TURNOVER = "turnover"
    TACKLE_WON = "tackle_won"
    FOUL_COMMITTED = "foul_committed"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    CORNER_AWARDED = "corner_awarded"
    FREE_KICK = "free_kick"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    SHOT_BLOCKED = "shot_blocked"
    SHOT_SAVED = "shot_saved"
    GOAL = "goal"
    SUBSTITUTION = "substitution"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"


@dataclass
class MatchEvent:
    """Individual discrete event occurring during a simulated match."""
    minute: int
    second: int
    team_id: int
    event_type: EventType
    zone: PitchZone
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    xg: float = 0.0
    description: str = ""
    score_home_after: int = 0
    score_away_after: int = 0


@dataclass
class MatchState:
    """
    Dynamic state of a simulated match at any point in time.
    Tracks clock, score, ball position, tactical tension, discipline, and stamina.
    """
    minute: int = 0
    second: int = 0
    home_team_id: int = 1
    away_team_id: int = 2
    score_h: int = 0
    score_a: int = 0
    possession_team_id: int = 0
    ball_zone: PitchZone = PitchZone.MIDDLE_THIRD
    red_cards_h: int = 0
    red_cards_a: int = 0
    yellow_cards_h: dict[int, int] = field(default_factory=dict)
    yellow_cards_a: dict[int, int] = field(default_factory=dict)
    player_stamina_h: dict[int, float] = field(default_factory=dict)
    player_stamina_a: dict[int, float] = field(default_factory=dict)
    substitutions_used_h: int = 0
    substitutions_used_a: int = 0
    active_lineup_h: list[int] = field(default_factory=list)
    active_lineup_a: list[int] = field(default_factory=list)

    @property
    def score_diff_home(self) -> int:
        return self.score_h - self.score_a

    @property
    def is_half_time(self) -> bool:
        return self.minute == 45 and self.second == 0

    @property
    def is_full_time(self) -> bool:
        return self.minute >= 90
