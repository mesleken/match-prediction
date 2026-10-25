"""Client for API-Football (v3.football.api-sports.io) with free-tier probing and quota protection."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional
import duckdb
import rapidfuzz.process

from slmodel.ingest.http import HttpClient
from slmodel.ingest.quota import QuotaManager

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
SUPER_LIG_LEAGUE_ID = 203  # API-Football Süper Lig League ID


class ApiFootballClient:
    """Manages API-Football queries, probing season entitlement, and logging results."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "data/warehouse.duckdb",
    ) -> None:
        self.db_path = db_path
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "").strip()
        self.http_client = HttpClient(db_path=db_path, min_interval_sec=0.5)
        self.quota_mgr = QuotaManager(db_path=db_path)

    def probe_account_entitlement(self) -> dict[str, Any]:
        """
        Probe account entitlement and check if the active season (2026) is available on the free tier.
        Guarantees safe execution with zero exceptions.
        """
        if not self.api_key:
            return {
                "status": "no_key",
                "message": "API-Football anahtarı yapılandırılmamış (.env içinde API_FOOTBALL_KEY yok). Manuel giriş modu devrede.",
                "season_locked": True,
                "quota_remaining": 0,
            }

        headers = {
            "x-apisports-key": self.api_key,
            "Accept": "application/json",
        }

        # Single probe request on current season injuries
        url = f"{API_FOOTBALL_BASE_URL}/injuries"
        params = {"league": SUPER_LIG_LEAGUE_ID, "season": 2026}

        status, text = self.http_client.get(
            url=url,
            provider="api_football",
            params=params,
            headers=headers,
            ttl_seconds=3600,
        )

        quota_state = self.quota_mgr.load_state()
        rem_quota = quota_state.get("api_football_remaining", 100)

        if status == 200:
            try:
                data = json.loads(text)
                errors = data.get("errors", {})
                if errors:
                    err_msg = str(errors)
                    if "plan" in err_msg.lower() or "season" in err_msg.lower():
                        return {
                            "status": "locked_by_plan",
                            "message": f"Ücretsiz plan 2026 sezonunu kilitliyor: {err_msg}",
                            "season_locked": True,
                            "quota_remaining": rem_quota,
                        }
                    return {
                        "status": "error",
                        "message": f"API Hatası: {err_msg}",
                        "season_locked": True,
                        "quota_remaining": rem_quota,
                    }

                injuries = data.get("response", [])
                return {
                    "status": "active",
                    "message": f"API-Football aktif. {len(injuries)} sakatlık/ceza kaydı mevcut.",
                    "season_locked": False,
                    "quota_remaining": rem_quota,
                    "count": len(injuries),
                }
            except Exception as e:
                return {
                    "status": "parse_error",
                    "message": f"Yanıt ayrıştırılamadı: {e}",
                    "season_locked": True,
                    "quota_remaining": rem_quota,
                }

        if status == 429:
            return {
                "status": "rate_limited",
                "message": "Günlük API kotası tükendi (100 çağrı/gün).",
                "season_locked": True,
                "quota_remaining": 0,
            }

        return {
            "status": "http_error",
            "message": f"HTTP {status} yanıtı alındı.",
            "season_locked": True,
            "quota_remaining": rem_quota,
        }

    def sync_injuries_to_snapshots(self, season_year: int = 2026) -> dict[str, Any]:
        """
        Fetch injuries for Süper Lig and persist into availability_snapshots.
        Fuzzy matches API player names with local players table.
        """
        probe = self.probe_account_entitlement()
        if probe.get("season_locked"):
            return probe

        headers = {
            "x-apisports-key": self.api_key,
            "Accept": "application/json",
        }
        url = f"{API_FOOTBALL_BASE_URL}/injuries"
        params = {"league": SUPER_LIG_LEAGUE_ID, "season": season_year}

        status, text = self.http_client.get(
            url=url,
            provider="api_football",
            params=params,
            headers=headers,
            ttl_seconds=14400,
        )
        if status != 200:
            return {"status": "error", "message": f"HTTP {status}"}

        data = json.loads(text)
        injuries = data.get("response", [])
        if not injuries:
            return {"status": "empty", "message": "Kayıt dönmedi"}

        con = duckdb.connect(self.db_path)
        synced = 0
        now_utc = datetime.now(timezone.utc)

        try:
            # Load local players for fuzzy matching
            players_df = con.execute("SELECT player_id, team_id, name FROM players").fetchdf()
            players_by_team: dict[int, list[tuple[int, str]]] = {}
            for _, r in players_df.iterrows():
                players_by_team.setdefault(int(r["team_id"]), []).append((int(r["player_id"]), str(r["name"])))

            # Canonical alias lookup for team IDs
            aliases_df = con.execute("SELECT source_name, team_id FROM team_aliases").fetchdf()
            alias_map = dict(zip(aliases_df["source_name"].str.lower(), aliases_df["team_id"]))

            for inj in injuries:
                team_obj = inj.get("team", {})
                player_obj = inj.get("player", {})
                api_team_name = team_obj.get("name", "").strip().lower()
                api_player_name = player_obj.get("name", "").strip()

                team_id = alias_map.get(api_team_name)
                player_id = None

                if team_id and team_id in players_by_team and api_player_name:
                    team_roster = players_by_team[team_id]
                    names = [p[1] for p in team_roster]
                    best = rapidfuzz.process.extractOne(api_player_name, names, score_cutoff=75)
                    if best:
                        match_idx = names.index(best[0])
                        player_id = team_roster[match_idx][0]

                if not team_id:
                    continue

                raw_status = str(inj.get("player", {}).get("type", "injured")).lower()
                status_mapped = "suspended" if "suspen" in raw_status else ("doubtful" if "doubt" in raw_status else "injured")
                reason = inj.get("player", {}).get("reason", "")
                snap_id = int(now_utc.timestamp() * 1000) + synced

                con.execute("""
                    INSERT OR REPLACE INTO availability_snapshots (
                        snapshot_id, team_id, player_id, player_name, status,
                        reason, expected_return, source, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'api_football', ?)
                """, [snap_id, team_id, player_id, api_player_name, status_mapped, reason, now_utc])
                synced += 1

        finally:
            con.close()

        return {"status": "success", "synced_injuries": synced}
