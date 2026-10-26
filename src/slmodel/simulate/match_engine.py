"""
Event-based 90-minute match simulation engine.
Executes sequence-by-sequence match flow, tracking timeline, stats, and game-state dynamics.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from slmodel.simulate.state import EventType, MatchEvent, MatchState, PitchZone
from slmodel.simulate.tactics import TacticalProfile
from slmodel.simulate.event_generator import EventGenerator
from slmodel.simulate.params import EngineParams, DEFAULT_ENGINE_PARAMS


@dataclass
class MatchStats:
    shots_h: int = 0
    shots_a: int = 0
    shots_on_target_h: int = 0
    shots_on_target_a: int = 0
    xg_h: float = 0.0
    xg_a: float = 0.0
    corners_h: int = 0
    corners_a: int = 0
    fouls_h: int = 0
    fouls_a: int = 0
    yellow_cards_h: int = 0
    yellow_cards_a: int = 0
    red_cards_h: int = 0
    red_cards_a: int = 0
    goal_minutes_h: list[int] = field(default_factory=list)
    goal_minutes_a: list[int] = field(default_factory=list)


@dataclass
class MatchTimeline:
    """Full chronological timeline and statistical recap of a simulated match."""
    home_team_id: int
    away_team_id: int
    score_h: int
    score_a: int
    ht_score_h: int
    ht_score_a: int
    stats: MatchStats
    events: list[MatchEvent]

    @property
    def outcome_1x2(self) -> str:
        if self.score_h > self.score_a:
            return "1"
        elif self.score_h == self.score_a:
            return "X"
        return "2"


class MatchEngine:
    """Simulates an entire 90-minute football match through discrete tactical events."""

    def __init__(
        self,
        rng: Optional[np.random.RandomState] = None,
        params: Optional[EngineParams] = None,
    ) -> None:
        self.rng = rng or np.random.RandomState()
        self.params = params or DEFAULT_ENGINE_PARAMS
        self.generator = EventGenerator(rng=self.rng, params=self.params)

    def simulate_match(
        self,
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
        seed: Optional[int] = None,
    ) -> MatchTimeline:
        """Simulate complete match from kickoff to final whistle."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.generator = EventGenerator(rng=self.rng, params=self.params)

        # Dynamic Dixon-Coles expected goal scaling
        if lam_h is not None and lam_a is not None:
            ref_h = self.params.league_mean_lambda
            ref_a = self.params.league_mean_mu
            k_h = float(np.log(max(lam_h, 0.1) / ref_h))
            k_a = float(np.log(max(lam_a, 0.1) / ref_a))
            att_h = float(np.clip(0.20 + 0.45 * k_h, 0.05, 0.45))
            def_a = float(np.clip(0.20 - 0.35 * k_h, 0.05, 0.45))
            att_a = float(np.clip(0.20 + 0.45 * k_a, 0.05, 0.45))
            def_h = float(np.clip(0.20 - 0.35 * k_a, 0.05, 0.45))

        state = MatchState(
            minute=0, second=0,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            score_h=0, score_a=0,
            possession_team_id=home_team_id if self.rng.rand() < 0.50 else away_team_id,
            ball_zone=PitchZone.MIDDLE_THIRD,
        )

        all_events: list[MatchEvent] = []
        ht_score_h = 0
        ht_score_a = 0
        half_time_processed = False

        stats = MatchStats()

        while not state.is_full_time:
            # Check half-time transition at min 45
            if state.minute >= 45 and not half_time_processed:
                ht_score_h = state.score_h
                ht_score_a = state.score_a
                half_time_processed = True
                all_events.append(MatchEvent(
                    minute=45, second=0, team_id=0,
                    event_type=EventType.HALF_TIME, zone=PitchZone.MIDDLE_THIRD,
                    description=f"İlk Yarı Sona Erdi: {state.score_h} - {state.score_a}",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
                # 2nd half kickoff
                state.possession_team_id = away_team_id if state.possession_team_id == home_team_id else home_team_id
                state.ball_zone = PitchZone.MIDDLE_THIRD

            seq_events = self.generator.advance_possession(
                state=state,
                team_h_id=home_team_id,
                team_a_id=away_team_id,
                att_rating_h=att_h,
                def_rating_h=def_h,
                att_rating_a=att_a,
                def_rating_a=def_a,
                base_tactics_h=tactics_h,
                base_tactics_a=tactics_a,
            )

            for ev in seq_events:
                all_events.append(ev)
                is_h = (ev.team_id == home_team_id)

                if ev.event_type == EventType.GOAL:
                    if is_h:
                        stats.xg_h += ev.xg
                        stats.shots_h += 1
                        stats.shots_on_target_h += 1
                        stats.goal_minutes_h.append(ev.minute)
                    else:
                        stats.xg_a += ev.xg
                        stats.shots_a += 1
                        stats.shots_on_target_a += 1
                        stats.goal_minutes_a.append(ev.minute)
                elif ev.event_type == EventType.SHOT_SAVED:
                    if is_h:
                        stats.xg_h += ev.xg
                        stats.shots_h += 1
                        stats.shots_on_target_h += 1
                    else:
                        stats.xg_a += ev.xg
                        stats.shots_a += 1
                        stats.shots_on_target_a += 1
                elif ev.event_type in (EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED):
                    if is_h:
                        stats.xg_h += ev.xg
                        stats.shots_h += 1
                    else:
                        stats.xg_a += ev.xg
                        stats.shots_a += 1
                elif ev.event_type == EventType.CORNER_AWARDED:
                    if is_h: stats.corners_h += 1
                    else: stats.corners_a += 1
                elif ev.event_type == EventType.FOUL_COMMITTED:
                    if is_h: stats.fouls_h += 1
                    else: stats.fouls_a += 1
                elif ev.event_type == EventType.YELLOW_CARD:
                    if is_h: stats.yellow_cards_h += 1
                    else: stats.yellow_cards_a += 1
                elif ev.event_type == EventType.RED_CARD:
                    if is_h: stats.red_cards_h += 1
                    else: stats.red_cards_a += 1

        stats.red_cards_h = state.red_cards_h
        stats.red_cards_a = state.red_cards_a

        return MatchTimeline(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            score_h=state.score_h,
            score_a=state.score_a,
            ht_score_h=ht_score_h,
            ht_score_a=ht_score_a,
            stats=stats,
            events=all_events,
        )
