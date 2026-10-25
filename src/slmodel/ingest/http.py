"""Shared robust HTTP client with disk caching, rate limiting, and quota logging."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Optional
import httpx

from slmodel.ingest.quota import QuotaManager

CACHE_BASE_DIR = Path("data/cache/raw")

DEFAULT_MACKOLIK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://arsiv.mackolik.com/",
    "Accept": "application/json, text/javascript, text/html, */*; q=0.01",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}

DEFAULT_OPENMETEO_HEADERS = {
    "User-Agent": "SuperLigModel/0.2.0 (Research & Sports Prediction Engine)",
    "Accept": "application/json",
}


class HttpClient:
    """Robust cached HTTP client with exponential backoff and quota tracking."""

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        min_interval_sec: float = 1.0,
    ) -> None:
        self.db_path = db_path
        self.min_interval_sec = min_interval_sec
        self.quota_mgr = QuotaManager(db_path=db_path)
        self._last_call_time: dict[str, float] = {}

    def _get_cache_path(self, provider: str, url: str, params: Optional[dict[str, Any]] = None) -> Path:
        key_raw = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        hashed = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:24]
        return CACHE_BASE_DIR / provider / f"{hashed}.json"

    def get(
        self,
        url: str,
        provider: str = "generic",
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        ttl_seconds: int = 86400,
        force_refresh: bool = False,
        timeout_sec: float = 12.0,
        max_retries: int = 3,
    ) -> tuple[int, str]:
        """
        Perform an HTTP GET request with disk caching, courteous throttling, and retries.
        Returns: (http_status_code, response_text).
        """
        cache_file = self._get_cache_path(provider, url, params)

        # 1. Check disk cache
        if not force_refresh and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                cached_time = cached.get("timestamp", 0)
                if (time.time() - cached_time) < ttl_seconds:
                    return cached.get("status", 200), cached.get("text", "")
            except Exception:
                pass

        # 2. Check quota for rate-limited providers (API-Football)
        if provider == "api_football" and not self.quota_mgr.can_request("api_football"):
            return 429, json.dumps({"error": "Quota budget exhausted or throttled for today"})

        # 3. Enforce min interval throttling per provider
        now = time.time()
        last_t = self._last_call_time.get(provider, 0.0)
        delay_needed = self.min_interval_sec - (now - last_t)
        if delay_needed > 0:
            time.sleep(delay_needed)
        self._last_call_time[provider] = time.time()

        # 4. Prepare headers
        req_headers = dict(headers or {})
        if provider == "mackolik" and not req_headers:
            req_headers = DEFAULT_MACKOLIK_HEADERS
        elif provider == "open_meteo" and not req_headers:
            req_headers = DEFAULT_OPENMETEO_HEADERS

        # 5. Execute request with exponential backoff
        status_code = 0
        response_text = ""
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
                    resp = client.get(url, params=params, headers=req_headers)
                    status_code = resp.status_code
                    response_text = resp.text

                # Parse API-Football remaining quota header if present
                quota_rem = None
                if "x-ratelimit-requests-remaining" in resp.headers:
                    try:
                        quota_rem = int(resp.headers["x-ratelimit-requests-remaining"])
                    except Exception:
                        pass

                self.quota_mgr.record_call(
                    provider=provider,
                    endpoint=url.split("?")[0],
                    status_code=status_code,
                    quota_remaining=quota_rem,
                    detail=f"Attempt {attempt + 1}, len={len(response_text)}",
                )

                if status_code in (200, 204):
                    # Cache successful response
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(
                        json.dumps({
                            "timestamp": time.time(),
                            "status": status_code,
                            "text": response_text,
                        }, ensure_ascii=False),
                        encoding="utf-8"
                    )
                    return status_code, response_text

                if status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                return status_code, response_text
            except Exception as e:
                if attempt == max_retries - 1:
                    response_text = str(e)
                    status_code = 599
                time.sleep(backoff)
                backoff *= 2.0

        return status_code, response_text
