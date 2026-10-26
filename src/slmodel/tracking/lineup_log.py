"""
Prospective Lineup & Availability Logging Pipeline.
Records verified pre-match snapshots with strict UTC timestamps (as_of) prior to kickoff.
"""

from datetime import datetime, timezone
from typing import Any, Optional
import duckdb
import pandas as pd

from slmodel.models.availability import STATUS_PROBABILITIES


def record_team_availability_snapshot(
    team_id: int,
    players_status: list[dict[str, Any]],
    source: str = "manual",
    as_of: Optional[datetime] = None,
    db_path: str = "data/warehouse.duckdb",
) -> int:
    """
    Record a timestamped availability snapshot for a club.
    players_status: list of dicts with keys: player_id, player_name, status ('available', 'doubtful', 'out'), reason
    """
    con = duckdb.connect(db_path)
    now_utc = as_of or datetime.now(timezone.utc)
    base_id = int(now_utc.timestamp() * 1000)

    inserted = 0
    try:
        for idx, p in enumerate(players_status):
            pid = p.get("player_id")
            p_name = p.get("player_name") or f"Player_{pid}"
            status = str(p.get("status", "available")).lower()
            reason = p.get("reason", "")
            exp_ret = p.get("expected_return", None)
            snap_id = base_id + idx

            con.execute("""
                INSERT INTO availability_snapshots (
                    snapshot_id, team_id, player_id, player_name, status,
                    reason, expected_return, source, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [snap_id, team_id, pid, p_name, status, reason, exp_ret, source, now_utc])
            inserted += 1
    finally:
        con.close()

    return inserted


def record_official_starting_xi(
    match_id: int,
    team_id: int,
    starter_player_ids: list[int],
    bench_player_ids: Optional[list[int]] = None,
    as_of: Optional[datetime] = None,
    db_path: str = "data/warehouse.duckdb",
) -> int:
    """
    Record verified starting XI and bench 1 hour before kickoff.
    Non-starters who are absent can be registered as out.
    """
    con = duckdb.connect(db_path)
    now_utc = as_of or datetime.now(timezone.utc)

    try:
        # Get full roster
        roster = con.execute("""
            SELECT player_id, name, is_starter FROM players WHERE team_id = ?
        """, [team_id]).fetchall()
    finally:
        con.close()

    starters_set = set(starter_player_ids)
    bench_set = set(bench_player_ids or [])

    snapshot_records = []
    for pid, name, is_base_starter in roster:
        if pid in starters_set:
            status = "available"
            reason = "İlk 11 Başlangıç Kadrosu"
        elif pid in bench_set:
            status = "available"
            reason = "Yedek Kulübesi"
        elif is_base_starter:
            # Regular starter missing from matchday sheet!
            status = "out"
            reason = "Maç Kadrosunda Yok (Rotasyon / Sakatlık)"
        else:
            status = "available"
            reason = "Kadro Dışı / Rezerv"

        snapshot_records.append({
            "player_id": pid,
            "player_name": name,
            "status": status,
            "reason": reason,
        })

    return record_team_availability_snapshot(
        team_id=team_id,
        players_status=snapshot_records,
        source="manual",
        as_of=now_utc,
        db_path=db_path,
    )


def get_latest_availability_summary(
    team_id: Optional[int] = None,
    db_path: str = "data/warehouse.duckdb",
) -> list[dict[str, Any]]:
    """Return latest recorded availability snapshots joined with team details."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        q = """
            SELECT s.snapshot_id, s.team_id, t.name as team_name,
                   s.player_id, s.player_name, s.status, s.reason,
                   s.source, s.fetched_at
            FROM availability_snapshots s
            JOIN teams t ON s.team_id = t.team_id
        """
        params = []
        if team_id is not None:
            q += " WHERE s.team_id = ?"
            params.append(team_id)
        q += " ORDER BY s.fetched_at DESC, s.team_id LIMIT 100"

        df = con.execute(q, params).fetchdf()
        return df.to_dict(orient="records")
    finally:
        con.close()
