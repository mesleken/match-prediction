"""Base HTTP fetcher with persistent disk caching and exponential backoff."""

import hashlib
import time
from pathlib import Path
from typing import Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

DEFAULT_TIMEOUT = 30.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

class BaseFetcher:
    """Handles cached, polite HTTP requests with retries."""

    def __init__(
        self,
        cache_dir: Path | str = "data/raw",
        request_delay_seconds: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay_seconds = request_delay_seconds
        self.headers = {"User-Agent": user_agent}
        self._last_request_time: float = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay_seconds:
            time.sleep(self.request_delay_seconds - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=2.0, max=10.0),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def fetch_url(
        self,
        url: str,
        cache_subpath: Optional[str] = None,
        force_refresh: bool = False,
    ) -> tuple[bytes, str]:
        """
        Fetch URL content.
        Returns:
            tuple[bytes, str]: (content_bytes, sha256_hash)
        """
        if cache_subpath:
            cached_file = self.cache_dir / cache_subpath
            if cached_file.exists() and not force_refresh:
                data = cached_file.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                return data, sha

        self._respect_rate_limit()
        with httpx.Client(headers=self.headers, timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content

        sha = hashlib.sha256(data).hexdigest()

        if cache_subpath:
            cached_file = self.cache_dir / cache_subpath
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(data)

        return data, sha
