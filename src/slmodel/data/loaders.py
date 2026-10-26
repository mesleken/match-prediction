"""
Unified, leakage-free match history and temporal data loader.
Guarantees parity between offline backtesting, evaluation scripts, and live production prediction pipelines.
"""

from datetime import datetime, timezone
from typing import Any, Optional
import duckdb
import pandas as pd


MATCH_HISTORY_QUERY = """
    SELECT 
        m.match_id, m.season, m.matchday, m.kickoff_utc,
        m.home_team_id, m.away_team_id,
        m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
        m.referee, m.stadium_id, m.status, m.crowd, m.is_derby,
        sh.shots as hs, sh.shots_on_target as hst, sh.corners as hc,
        sh.fouls as hf, sh.yellow as hy, sh.red as hr,
        sh.xg as hxg,
        sa.shots as as_, sa.shots_on_target as ast, sa.corners as ac,
        sa.fouls as af, sa.yellow as ay, sa.red as ar,
        sa.xg as axg
    FROM matches m
    LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
    LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
    WHERE m.status = 'played' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
      AND (cast(? as TIMESTAMP) IS NULL OR m.kickoff_utc <= cast(? as TIMESTAMP))
    ORDER BY m.kickoff_utc, m.match_id
"""


def load_match_history(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: str = "data/warehouse.duckdb",
    as_of: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load match history with consistent schema across all services, scripts, and tests.
    Strictly filters matches with kickoff_utc <= as_of when as_of is provided.
    """
    should_close = False
    if con is None:
        con = duckdb.connect(db_path, read_only=True)
        should_close = True

    try:
        as_of_dt = None
        if as_of is not None:
            as_of_dt = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        df = con.execute(MATCH_HISTORY_QUERY, [as_of_dt, as_of_dt]).fetchdf()
        return df
    finally:
        if should_close:
            con.close()
