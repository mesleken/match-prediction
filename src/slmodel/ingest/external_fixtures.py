"""Automatic non-league, European, and Domestic Cup fixture ingestion for Süper Lig clubs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Optional
import duckdb

from slmodel.ingest.fixtures import MACKOLIK_TO_WAREHOUSE, WAREHOUSE_NAMES
from slmodel.ingest.http import HttpClient

# Known European & Domestic Cup competition seasons in Mackolik (2026/27)
EXTERNAL_COMPETITIONS: list[dict[str, Any]] = [
    {"season_id": 73968, "name": "UEFA Champions League"},
    {"season_id": 73989, "name": "UEFA Europa League"},
    {"season_id": 74012, "name": "Turkiye Kupasi"},
    {"season_id": 73802, "name": "UEFA Champions League Play-Off"},
    {"season_id": 73894, "name": "UEFA Europa League Play-Off"},
]


class ExternalFixturesClient:
    """Fetches non-league domestic and European fixtures from Mackolik and API-Football."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = db_path
        self.http_client = HttpClient(db_path=db_path, min_interval_sec=0.4)

    def fetch_competition_weeks(self, season_id: int) -> list[dict[str, Any]]:
        """Fetch all fixture weeks for a given competition season."""
        url = f"https://arsiv.mackolik.com/AjaxHandlers/FixtureHandler.aspx?command=getWeeks&id={season_id}"
        status, text = self.http_client.get(url=url, provider="mackolik", ttl_seconds=86400 * 2)
        if status != 200:
            return []
        try:
            raw = json.loads(text.replace("'", '"'))
            weeks = []
            for w in raw:
                weeks.append({
                    "week": int(w[0]),
                    "start_date": str(w[1]),
                    "end_date": str(w[2]),
                    "is_finished": bool(w[3] == 1),
                })
            return weeks
        except Exception:
            return []

    def fetch_competition_week_matches(self, season_id: int, week: int) -> list[dict[str, Any]]:
        """Fetch scheduled/played matches for a specific competition week."""
        url = f"https://arsiv.mackolik.com/AjaxHandlers/FixtureHandler.aspx?command=getMatches&id={season_id}&week={week}"
        status, text = self.http_client.get(url=url, provider="mackolik", ttl_seconds=86400)
        if status != 200:
            return []
        try:
            cleaned = text.replace("'", '"')
            cleaned = re.sub(r",\s*,", ", null,", cleaned)
            cleaned = re.sub(r",\s*,", ", null,", cleaned)
            data = json.loads(cleaned)
            matches = []
            for m in data:
                if isinstance(m, list) and len(m) >= 7:
                    matches.append({
                        "mackolik_match_id": m[0],
                        "date_str": str(m[1]),
                        "status_str": str(m[2]),
                        "home_mid": m[3],
                        "home_name": str(m[4]),
                        "away_mid": m[5],
                        "away_name": str(m[6]),
                    })
            return matches
        except Exception:
            return []

    def sync_all_external_fixtures(self) -> dict[str, Any]:
        """
        Scan all external European and domestic cup competitions,
        extract matches involving any Süper Lig team, and upsert to external_fixtures table.
        """
        con = duckdb.connect(self.db_path)
        now_utc = datetime.now(timezone.utc)
        synced_count = 0

        try:
            for comp in EXTERNAL_COMPETITIONS:
                sid = comp["season_id"]
                c_name = comp["name"]
                weeks = self.fetch_competition_weeks(sid)

                for w_info in weeks:
                    w_num = w_info["week"]
                    w_matches = self.fetch_competition_week_matches(sid, w_num)

                    for m in w_matches:
                        h_mid = m.get("home_mid")
                        a_mid = m.get("away_mid")
                        h_wid = MACKOLIK_TO_WAREHOUSE.get(h_mid)
                        a_wid = MACKOLIK_TO_WAREHOUSE.get(a_mid)

                        # Match involves at least one Süper Lig team
                        if h_wid or a_wid:
                            mk_id = m["mackolik_match_id"]
                            d_str = m.get("date_str", "")
                            # Parse approximate kickoff date (e.g. '08/09' -> 2026-09-08)
                            try:
                                parts = d_str.split("/")
                                day = int(parts[0])
                                month = int(parts[1])
                                year = 2026 if month >= 6 else 2027
                                kickoff_dt = datetime(year, month, day, 19, 0, tzinfo=timezone.utc)
                            except Exception:
                                kickoff_dt = now_utc

                            # Insert for home team if Turkish
                            if h_wid:
                                ext_id = f"mk_{mk_id}_h_{h_wid}"
                                is_home = True
                                opponent = m.get("away_name", "Unknown")
                                stat = "played" if m.get("status_str") == "MS" else "scheduled"
                                con.execute("""
                                    INSERT OR REPLACE INTO external_fixtures (
                                        ext_id, team_id, competition, competition_season_id,
                                        kickoff_utc, opponent, is_home, status, source, fetched_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'mackolik', ?)
                                """, [ext_id, h_wid, c_name, sid, kickoff_dt, opponent, is_home, stat, now_utc])
                                synced_count += 1

                            # Insert for away team if Turkish
                            if a_wid:
                                ext_id = f"mk_{mk_id}_a_{a_wid}"
                                is_home = False
                                opponent = m.get("home_name", "Unknown")
                                stat = "played" if m.get("status_str") == "MS" else "scheduled"
                                con.execute("""
                                    INSERT OR REPLACE INTO external_fixtures (
                                        ext_id, team_id, competition, competition_season_id,
                                        kickoff_utc, opponent, is_home, status, source, fetched_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'mackolik', ?)
                                """, [ext_id, a_wid, c_name, sid, kickoff_dt, opponent, is_home, stat, now_utc])
                                synced_count += 1

        finally:
            con.close()

        return {
            "status": "success",
            "synced_external_fixtures": synced_count,
            "competitions_tracked": len(EXTERNAL_COMPETITIONS),
        }
