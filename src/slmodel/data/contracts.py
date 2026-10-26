"""
Temporal data contracts and validation boundaries for match prediction engine.
Strictly prevents lookahead leakage by recording when data was observed vs event time.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class OddsSnapshotKind(str, Enum):
    OPENING = "open"
    T_24H = "t24"
    T_6H = "t6"
    T_1H = "t1"
    CLOSING = "closing"


@dataclass(frozen=True)
class TemporalMetadata:
    """
    Temporal audit metadata ensuring every fact known to the system is timestamped.
    """
    event_time: datetime
    observed_at: datetime
    source: str
    source_record_id: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revision: int = 1

    def is_observable_at(self, as_of: datetime) -> bool:
        """Return True if this information was strictly known prior to or at as_of."""
        as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        obs_utc = self.observed_at if self.observed_at.tzinfo else self.observed_at.replace(tzinfo=timezone.utc)
        if obs_utc > as_of_utc:
            return False
        if self.valid_from and (self.valid_from if self.valid_from.tzinfo else self.valid_from.replace(tzinfo=timezone.utc)) > as_of_utc:
            return False
        if self.valid_to and (self.valid_to if self.valid_to.tzinfo else self.valid_to.replace(tzinfo=timezone.utc)) <= as_of_utc:
            return False
        return True


@dataclass
class MatchDataContract:
    """Verified match contract prior to model ingestion."""
    match_id: int
    season: str
    matchday: int
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    ht_home_goals: Optional[int] = None
    ht_away_goals: Optional[int] = None
    referee: Optional[str] = None
    stadium_id: Optional[int] = None
    status: str = "played"
    crowd: str = "full"
    is_derby: bool = False
    temporal: Optional[TemporalMetadata] = None


@dataclass
class LineupSnapshotContract:
    """Lineup observation contract with explicit observation timing."""
    snapshot_id: str
    team_id: int
    match_id: int
    observed_at: datetime
    is_official: bool  # True if confirmed starting XI (T-1h), False if predicted/typical XI
    starting_player_ids: list[int]
    bench_player_ids: list[int]
    source: str
    notes: Optional[str] = None

    def is_observable_at(self, as_of: datetime) -> bool:
        as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        obs_utc = self.observed_at if self.observed_at.tzinfo else self.observed_at.replace(tzinfo=timezone.utc)
        return obs_utc <= as_of_utc


@dataclass
class OddsSnapshotContract:
    """Timestamped betting odds contract."""
    snapshot_id: int
    match_id: int
    bookmaker: str
    kind: OddsSnapshotKind
    market: str  # '1x2', 'ou', 'ah', 'btts'
    selection: str  # '1', 'X', '2', 'over', 'under', 'yes', 'no'
    odds: float
    taken_at: datetime
    source: str
    line: Optional[float] = None

    def is_observable_at(self, as_of: datetime) -> bool:
        as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        taken_utc = self.taken_at if self.taken_at.tzinfo else self.taken_at.replace(tzinfo=timezone.utc)
        return taken_utc <= as_of_utc
