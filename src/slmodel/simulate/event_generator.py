"""
Semi-Markov event transition generator for full-match possession simulation.
"""

from typing import Any, Optional
import numpy as np

from slmodel.simulate.state import EventType, MatchEvent, MatchState, PitchZone
from slmodel.simulate.tactics import TacticalProfile, adjust_tactics_for_game_state
from slmodel.simulate.shot_model import resolve_shot_attempt
from slmodel.simulate.params import EngineParams, DEFAULT_ENGINE_PARAMS


class EventGenerator:
    """
    Generates and resolves possession sequence events conditioned on match state,
    tactical profiles, player quality differentials, and spatial pitch zones.
    """

    def __init__(
        self,
        rng: Optional[np.random.RandomState] = None,
        params: Optional[EngineParams] = None,
    ) -> None:
        self.rng = rng or np.random.RandomState()
        self.params = params or DEFAULT_ENGINE_PARAMS

    def advance_possession(
        self,
        state: MatchState,
        team_h_id: int,
        team_a_id: int,
        att_rating_h: float = 0.20,
        def_rating_h: float = 0.20,
        att_rating_a: float = 0.20,
        def_rating_a: float = 0.20,
        base_tactics_h: Optional[TacticalProfile] = None,
        base_tactics_a: Optional[TacticalProfile] = None,
    ) -> list[MatchEvent]:
        """
        Advance the match by one discrete possession phase (typically 18-36 seconds).
        Returns list of events produced during the sequence.
        """
        events: list[MatchEvent] = []
        is_home_possession = (state.possession_team_id == team_h_id)
        active_team_id = team_h_id if is_home_possession else team_a_id
        defending_team_id = team_a_id if is_home_possession else team_h_id

        # Tactical adjustments
        tactics_h = adjust_tactics_for_game_state(
            base_tactics_h or TacticalProfile(),
            score_diff=state.score_diff_home,
            minute=state.minute,
            own_red_cards=state.red_cards_h,
            opp_red_cards=state.red_cards_a,
        )
        tactics_a = adjust_tactics_for_game_state(
            base_tactics_a or TacticalProfile(),
            score_diff=-state.score_diff_home,
            minute=state.minute,
            own_red_cards=state.red_cards_a,
            opp_red_cards=state.red_cards_h,
        )

        active_tactics = tactics_h if is_home_possession else tactics_a
        defending_tactics = tactics_a if is_home_possession else tactics_h
        active_att = att_rating_h if is_home_possession else att_rating_a
        defending_def = def_rating_a if is_home_possession else def_rating_h

        # Advance clock by calibrated micro-possession seconds
        min_s = self.params.min_sequence_sec
        max_s = self.params.max_sequence_sec
        elapsed_sec = int(self.rng.randint(min_s, max_s) / max(active_tactics.tempo, 0.7))
        state.second += elapsed_sec
        while state.second >= 60:
            state.minute += 1
            state.second -= 60

        if state.is_full_time:
            events.append(MatchEvent(
                minute=90, second=0, team_id=0,
                event_type=EventType.FULL_TIME, zone=state.ball_zone,
                description="Maç Sona Erdi",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))
            return events

        # Substitution trigger around min 60-75
        if state.minute in (62, 68, 74) and state.second < 25:
            if is_home_possession and state.substitutions_used_h < 5:
                state.substitutions_used_h += 1
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=team_h_id,
                    event_type=EventType.SUBSTITUTION, zone=PitchZone.MIDDLE_THIRD,
                    description="Ev Sahibi Oyuncu Değişikliği (Taktik Tazeleme)",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            elif not is_home_possession and state.substitutions_used_a < 5:
                state.substitutions_used_a += 1
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=team_a_id,
                    event_type=EventType.SUBSTITUTION, zone=PitchZone.MIDDLE_THIRD,
                    description="Deplasman Oyuncu Değişikliği (Taktik Tazeleme)",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))

        # Zone-based state machine
        if state.ball_zone == PitchZone.DEFENSIVE_THIRD:
            p_turnover = self.params.p_def_turnover_base * defending_tactics.pressing_intensity
            p_foul = 0.04
            roll = self.rng.rand()
            if roll < p_turnover:
                state.ball_zone = PitchZone.ATTACKING_THIRD
                state.possession_team_id = defending_team_id
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=defending_team_id,
                    event_type=EventType.TURNOVER, zone=PitchZone.ATTACKING_THIRD,
                    description="Ön Alanda Baskı Sonucu Top Kazanımı",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            elif roll < p_turnover + p_foul:
                self._handle_foul(state, defending_team_id, active_team_id, events)
            elif roll < self.params.p_def_pass_middle:
                state.ball_zone = PitchZone.MIDDLE_THIRD
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=active_team_id,
                    event_type=EventType.PASS_COMPLETED, zone=PitchZone.MIDDLE_THIRD,
                    description="Geriden Oyun Kurulumu: Orta Sahaya Geçiş",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            else:
                # Retention in defensive third
                pass

        elif state.ball_zone == PitchZone.MIDDLE_THIRD:
            roll = self.rng.rand()
            diff = (active_att - defending_def)
            p_advance = self.params.p_mid_advance_base + float(np.clip(diff * 0.28, -0.15, 0.15)) + (active_tactics.risk_appetite - 0.5) * 0.06
            p_turnover = self.params.p_mid_turnover_base + (defending_tactics.pressing_intensity - 0.5) * 0.06
            p_foul = self.params.p_mid_foul_base

            if roll < p_advance:
                state.ball_zone = PitchZone.ATTACKING_THIRD
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=active_team_id,
                    event_type=EventType.PROGRESSIVE_CARRY, zone=PitchZone.ATTACKING_THIRD,
                    description="Orta Saha Kırıcı Pas ve Hücum Bölgesine İlerleme",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            elif roll < p_advance + p_turnover:
                state.possession_team_id = defending_team_id
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=defending_team_id,
                    event_type=EventType.TACKLE_WON, zone=PitchZone.MIDDLE_THIRD,
                    description="Orta Sahada İkili Mücadele ve Top Kaybı",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            elif roll < p_advance + p_turnover + p_foul:
                self._handle_foul(state, defending_team_id, active_team_id, events)
            else:
                # Retention / sideways circulation in middle third
                pass

        elif state.ball_zone == PitchZone.ATTACKING_THIRD:
            roll = self.rng.rand()
            p_box = self.params.p_att_box_base + (active_tactics.risk_appetite - 0.5) * 0.06
            p_direct_shot = self.params.p_att_shot_base + (active_att - 0.20) * 0.12
            p_corner = self.params.p_att_corner_base
            p_foul = self.params.p_att_foul_base
            p_recycle = self.params.p_att_recycle_base

            if roll < p_box:
                state.ball_zone = PitchZone.PENALTY_BOX
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=active_team_id,
                    event_type=EventType.PASS_COMPLETED, zone=PitchZone.PENALTY_BOX,
                    description="Ceza Sahasına Tehlikeli Ara Pas / İç Koridor Girişi",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            elif roll < p_box + p_direct_shot:
                # Long distance shot
                res = resolve_shot_attempt(
                    PitchZone.ATTACKING_THIRD,
                    shooter_att=active_att,
                    gk_def=defending_def,
                    pressure=defending_tactics.line_height,
                    rng=self.rng,
                )
                self._handle_shot_resolution(state, res, active_team_id, defending_team_id, PitchZone.ATTACKING_THIRD, events)
            elif roll < p_box + p_direct_shot + p_corner:
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=active_team_id,
                    event_type=EventType.CORNER_AWARDED, zone=PitchZone.PENALTY_BOX,
                    description="Savunmaya Çarpan Top Korner",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
                state.ball_zone = PitchZone.PENALTY_BOX
            elif roll < p_box + p_direct_shot + p_corner + p_foul:
                self._handle_foul(state, defending_team_id, active_team_id, events)
            elif roll < p_box + p_direct_shot + p_corner + p_foul + p_recycle:
                # Recycle back to midfield
                state.ball_zone = PitchZone.MIDDLE_THIRD
            else:
                # Turnover
                state.possession_team_id = defending_team_id
                state.ball_zone = PitchZone.DEFENSIVE_THIRD

        elif state.ball_zone == PitchZone.PENALTY_BOX:
            # High-intensity goalmouth event
            roll = self.rng.rand()
            p_shot = self.params.p_box_shot_base
            p_corner = self.params.p_box_corner_base
            p_foul = self.params.p_box_foul_base

            if roll < p_shot:
                res = resolve_shot_attempt(
                    PitchZone.PENALTY_BOX,
                    shooter_att=active_att,
                    gk_def=defending_def,
                    pressure=defending_tactics.line_height,
                    rng=self.rng,
                )
                self._handle_shot_resolution(state, res, active_team_id, defending_team_id, PitchZone.PENALTY_BOX, events)
            elif roll < p_shot + p_corner:
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=active_team_id,
                    event_type=EventType.CORNER_AWARDED, zone=PitchZone.PENALTY_BOX,
                    description="Altıpas Karambolü ve Savunma Müdahalesi: Korner",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
            elif roll < p_shot + p_corner + p_foul:
                self._handle_foul(state, defending_team_id, active_team_id, events, is_penalty=True)
            else:
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=defending_team_id,
                    event_type=EventType.TACKLE_WON, zone=PitchZone.PENALTY_BOX,
                    description="Ceza Sahasında Kritik Savunma Uzaklaştırması",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
                state.possession_team_id = defending_team_id
                state.ball_zone = PitchZone.MIDDLE_THIRD

        return events

    def _handle_shot_resolution(
        self,
        state: MatchState,
        res: Any,
        att_team_id: int,
        def_team_id: int,
        zone: PitchZone,
        events: list[MatchEvent],
    ) -> None:
        if res.is_goal:
            if att_team_id == state.home_team_id:
                state.score_h += 1
            else:
                state.score_a += 1

            events.append(MatchEvent(
                minute=state.minute, second=state.second, team_id=att_team_id,
                event_type=EventType.GOAL, zone=zone, xg=res.xg,
                description=f"GOL! Ceza sahası vuruşu ağlarla buluştu (xG: {res.xg:.2f})",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))
            # Reset to center kick
            state.possession_team_id = def_team_id
            state.ball_zone = PitchZone.MIDDLE_THIRD
        elif res.is_saved:
            events.append(MatchEvent(
                minute=state.minute, second=state.second, team_id=att_team_id,
                event_type=EventType.SHOT_SAVED, zone=zone, xg=res.xg,
                description=f"Kaleci Kurtarışı! İsabetli şut kalecide kaldı (xG: {res.xg:.2f})",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))
            if self.rng.rand() < self.params.save_corner_rebound_prob:
                events.append(MatchEvent(
                    minute=state.minute, second=state.second, team_id=att_team_id,
                    event_type=EventType.CORNER_AWARDED, zone=PitchZone.PENALTY_BOX,
                    description="Kaleciden Seken Top Korner",
                    score_home_after=state.score_h, score_away_after=state.score_a
                ))
                state.ball_zone = PitchZone.PENALTY_BOX
            else:
                state.possession_team_id = def_team_id
                state.ball_zone = PitchZone.DEFENSIVE_THIRD
        elif res.is_blocked:
            events.append(MatchEvent(
                minute=state.minute, second=state.second, team_id=att_team_id,
                event_type=EventType.SHOT_BLOCKED, zone=zone, xg=res.xg,
                description="Savunma Siper Oldu! Bloklanan şut",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))
            state.possession_team_id = def_team_id
            state.ball_zone = PitchZone.DEFENSIVE_THIRD
        else:
            events.append(MatchEvent(
                minute=state.minute, second=state.second, team_id=att_team_id,
                event_type=EventType.SHOT_OFF_TARGET, zone=zone, xg=res.xg,
                description=f"Şut Auta Çıktı (xG: {res.xg:.2f})",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))
            state.possession_team_id = def_team_id
            state.ball_zone = PitchZone.DEFENSIVE_THIRD

    def _handle_foul(
        self,
        state: MatchState,
        foul_team_id: int,
        drawn_team_id: int,
        events: list[MatchEvent],
        is_penalty: bool = False,
    ) -> None:
        p_yellow = self.params.p_foul_yellow
        p_red = self.params.p_foul_red

        events.append(MatchEvent(
            minute=state.minute, second=state.second, team_id=foul_team_id,
            event_type=EventType.FOUL_COMMITTED, zone=state.ball_zone,
            description="Faul düdüğü" if not is_penalty else "PENALTI! Ceza sahasında faul",
            score_home_after=state.score_h, score_away_after=state.score_a
        ))

        card_roll = self.rng.rand()
        if card_roll < p_red:
            # Direct Red
            if foul_team_id == state.home_team_id:
                state.red_cards_h += 1
            else:
                state.red_cards_a += 1
            events.append(MatchEvent(
                minute=state.minute, second=state.second, team_id=foul_team_id,
                event_type=EventType.RED_CARD, zone=state.ball_zone,
                description="KIRMIZI KART! Hakem doğrudan ihraç kararı verdi",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))
        elif card_roll < p_red + p_yellow:
            events.append(MatchEvent(
                minute=state.minute, second=state.second, team_id=foul_team_id,
                event_type=EventType.YELLOW_CARD, zone=state.ball_zone,
                description="Sarı Kart!",
                score_home_after=state.score_h, score_away_after=state.score_a
            ))

        if is_penalty:
            # Penalty shot
            pen_res = resolve_shot_attempt(PitchZone.PENALTY_BOX, shooter_att=0.35, is_set_piece=True, rng=self.rng)
            self._handle_shot_resolution(state, pen_res, drawn_team_id, foul_team_id, PitchZone.PENALTY_BOX, events)
        else:
            state.possession_team_id = drawn_team_id
