"""Hybrid player availability resolver (Manual Overrides > API Snapshots > Derived > Baseline)."""

from datetime import datetime, timezone
from typing import Any, Optional
import duckdb
import pandas as pd

from slmodel.config import get_model_config

STATUS_PROBABILITIES: dict[str, float] = {
    "out": 0.0,
    "injured": 0.0,
    "suspended": 0.0,
    "doubtful": 0.50,
    "available": 1.0,
}


class AvailabilityResolver:
    """Resolves player availability and expected play probabilities with strict precedence."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = db_path
        self.cfg = get_model_config()

    def set_manual_override(
        self,
        team_id: int,
        player_id: int,
        status: str,
        match_id: Optional[int] = None,
        note: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> int:
        """Add or update a user-specified manual absence override."""
        status_norm = status.lower()
        if status_norm not in ("out", "doubtful", "available"):
            raise ValueError(f"Invalid status: {status}. Must be 'out', 'doubtful', or 'available'.")

        con = duckdb.connect(self.db_path)
        now_utc = datetime.now(timezone.utc)
        override_id = int(now_utc.timestamp() * 1000)

        try:
            # Deactivate existing active overrides for this player/match
            if match_id:
                con.execute("""
                    UPDATE availability_overrides
                    SET active = FALSE
                    WHERE player_id = ? AND match_id = ? AND active = TRUE
                """, [player_id, match_id])
            else:
                con.execute("""
                    UPDATE availability_overrides
                    SET active = FALSE
                    WHERE player_id = ? AND active = TRUE
                """, [player_id])

            con.execute("""
                INSERT INTO availability_overrides (
                    override_id, team_id, player_id, status, match_id, note,
                    created_at, expires_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            """, [override_id, team_id, player_id, status_norm, match_id, note, now_utc, expires_at])
        finally:
            con.close()

        return override_id

    def clear_manual_override(self, override_id: int) -> bool:
        """Deactivate a specific manual override by its ID."""
        con = duckdb.connect(self.db_path)
        try:
            con.execute("UPDATE availability_overrides SET active = FALSE WHERE override_id = ?", [override_id])
            return True
        except Exception:
            return False
        finally:
            con.close()

    def list_active_overrides(self, team_id: Optional[int] = None) -> list[dict[str, Any]]:
        """List currently active manual overrides, joined with player and team names."""
        con = duckdb.connect(self.db_path, read_only=True)
        now_utc = datetime.now(timezone.utc)
        try:
            q = """
                SELECT o.override_id, o.team_id, t.name as team_name,
                       o.player_id, p.name as player_name, p.position,
                       o.status, o.match_id, o.note, o.created_at, o.expires_at
                FROM availability_overrides o
                JOIN players p ON o.player_id = p.player_id
                JOIN teams t ON o.team_id = t.team_id
                WHERE o.active = TRUE AND (o.expires_at IS NULL OR o.expires_at > ?)
            """
            params: list[Any] = [now_utc]
            if team_id is not None:
                q += " AND o.team_id = ?"
                params.append(team_id)
            q += " ORDER BY o.created_at DESC"

            df = con.execute(q, params).fetchdf()
            return df.to_dict(orient="records")
        finally:
            con.close()

    def resolve_team_availability(
        self,
        team_id: int,
        match_id: Optional[int] = None,
        as_of: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """
        Resolve availability for all roster players of a team.
        Precedence: Manual Override > API Snapshot > Derived > Available.
        Strict time leakage: Only considers snapshots with fetched_at < as_of.
        """
        con = duckdb.connect(self.db_path, read_only=True)
        as_of_ts = as_of or datetime.now(timezone.utc)

        try:
            # 1. Roster
            players_df = con.execute("""
                SELECT player_id, team_id, name, position, age, att_rating, def_rating, overall_rating, is_starter
                FROM players
                WHERE team_id = ?
                ORDER BY is_starter DESC, overall_rating DESC
            """, [team_id]).fetchdf()

            if players_df.empty:
                return []

            # 2. Active manual overrides
            q_man = """
                SELECT player_id, status, note, created_at
                FROM availability_overrides
                WHERE team_id = ? AND active = TRUE
                  AND created_at <= ?
                  AND (expires_at IS NULL OR expires_at > ?)
            """
            man_params: list[Any] = [team_id, as_of_ts, as_of_ts]
            if match_id is not None:
                q_man += " AND (match_id IS NULL OR match_id = ?)"
                man_params.append(match_id)

            man_df = con.execute(q_man, man_params).fetchdf()
            man_overrides = {int(r["player_id"]): r for _, r in man_df.iterrows()}

            # 3. API snapshots strictly fetched prior to as_of
            q_snap = """
                SELECT player_id, status, reason, fetched_at
                FROM availability_snapshots
                WHERE team_id = ? AND fetched_at <= ? AND player_id IS NOT NULL
                ORDER BY fetched_at DESC
            """
            snap_df = con.execute(q_snap, [team_id, as_of_ts]).fetchdf()
            api_snapshots: dict[int, Any] = {}
            for _, r in snap_df.iterrows():
                pid = int(r["player_id"])
                if pid not in api_snapshots:
                    api_snapshots[pid] = r

            # 4. Resolve each player
            resolved = []
            for _, p in players_df.iterrows():
                pid = int(p["player_id"])
                p_name = str(p["name"])
                pos = str(p["position"])
                is_start = bool(p["is_starter"])

                status = "available"
                source = "baseline"
                reason = ""
                p_play = 1.0

                if pid in man_overrides:
                    # Tier 1: Manual Override
                    m_row = man_overrides[pid]
                    status = str(m_row["status"])
                    source = "manual"
                    reason = str(m_row["note"] or "Manuel kullanıcı girişi")
                    p_play = STATUS_PROBABILITIES.get(status, 1.0)
                elif pid in api_snapshots:
                    # Tier 2: API Snapshot
                    s_row = api_snapshots[pid]
                    status = str(s_row["status"])
                    source = "api_football"
                    reason = str(s_row["reason"] or "API-Football sakatlık listesi")
                    p_play = STATUS_PROBABILITIES.get(status, 0.0)

                resolved.append({
                    "player_id": pid,
                    "team_id": team_id,
                    "name": p_name,
                    "position": pos,
                    "is_starter": is_start,
                    "att_rating": float(p["att_rating"]),
                    "def_rating": float(p["def_rating"]),
                    "overall_rating": float(p["overall_rating"]),
                    "status": status,
                    "source": source,
                    "reason": reason,
                    "p_play": p_play,
                })

            return resolved
        finally:
            con.close()
