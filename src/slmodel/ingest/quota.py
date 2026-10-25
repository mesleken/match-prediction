"""API Quota and rate-limiting manager for third-party providers (API-Football, etc.)."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import duckdb

QUOTA_CACHE_PATH = "data/cache/quota_state.json"
API_FOOTBALL_DAILY_LIMIT = 100


class QuotaManager:
    """Tracks daily API calls and enforces limits to prevent account suspension."""

    def __init__(
        self,
        cache_path: str = QUOTA_CACHE_PATH,
        db_path: str = "data/warehouse.duckdb",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.db_path = db_path

    def _today_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def load_state(self) -> dict[str, Any]:
        """Load quota state from disk cache, resetting if a new UTC day has started."""
        today = self._today_utc()
        state = {
            "date": today,
            "api_football_used": 0,
            "api_football_limit": API_FOOTBALL_DAILY_LIMIT,
            "api_football_remaining": API_FOOTBALL_DAILY_LIMIT,
            "last_request_utc": None,
            "status": "active",
        }
        if self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if loaded.get("date") == today:
                    return loaded
            except Exception:
                pass
        self.save_state(state)
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        """Persist quota state to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def can_request(self, provider: str = "api_football") -> bool:
        """Check if an API call can be safely dispatched without exceeding budget."""
        if provider != "api_football":
            return True
        state = self.load_state()
        used = state.get("api_football_used", 0)
        limit = state.get("api_football_limit", API_FOOTBALL_DAILY_LIMIT)
        # Safety buffer at 80% usage (leave 20 requests for emergencies)
        if used >= int(limit * 0.85):
            return False
        return True

    def record_call(
        self,
        provider: str,
        endpoint: str,
        status_code: int,
        quota_remaining: Optional[int] = None,
        detail: str = "",
    ) -> dict[str, Any]:
        """Record an API request to disk state and write an entry into DuckDB ingest_log."""
        state = self.load_state()
        if provider == "api_football":
            state["api_football_used"] = state.get("api_football_used", 0) + 1
            if quota_remaining is not None:
                state["api_football_remaining"] = quota_remaining
            else:
                state["api_football_remaining"] = max(0, state["api_football_limit"] - state["api_football_used"])
            state["last_request_utc"] = datetime.now(timezone.utc).isoformat()
            if state["api_football_used"] >= state["api_football_limit"]:
                state["status"] = "exhausted"
            elif state["api_football_used"] >= int(state["api_football_limit"] * 0.85):
                state["status"] = "throttled"
            else:
                state["status"] = "active"
            self.save_state(state)

        # Log to DuckDB ingest_log table
        try:
            con = duckdb.connect(self.db_path)
            log_id = int(datetime.now(timezone.utc).timestamp() * 1000)
            now_iso = datetime.now(timezone.utc)
            con.execute("""
                INSERT OR REPLACE INTO ingest_log (
                    log_id, source, endpoint, requested_at, http_status, quota_remaining, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                log_id, provider, endpoint, now_iso, status_code,
                state.get("api_football_remaining", 0) if provider == "api_football" else None,
                detail[:500] if detail else None,
            ])
            con.close()
        except Exception:
            pass

        return state

    def get_status_summary(self) -> dict[str, Any]:
        """Get human-readable quota status for UI display."""
        state = self.load_state()
        return {
            "provider": "api_football",
            "date": state.get("date"),
            "used": state.get("api_football_used", 0),
            "remaining": state.get("api_football_remaining", API_FOOTBALL_DAILY_LIMIT),
            "limit": state.get("api_football_limit", API_FOOTBALL_DAILY_LIMIT),
            "status": state.get("status", "active"),
            "last_request": state.get("last_request_utc"),
        }
