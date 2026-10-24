"""Data synchronization module to fetch latest match stats, fixtures, and odds into DuckDB."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import yaml

from slmodel.ingest.footballdata import FootballDataFetcher
from slmodel.etl.transform import WarehouseETL


def sync_latest_data(
    seasons: Optional[list[str]] = None,
    league: str = "T1",
    sync_fixtures: bool = True,
    meta_path: str = "data/sync_meta.json",
) -> dict[str, Any]:
    """
    Fetch the latest raw match CSV files from football-data.co.uk with cache bypass (force_refresh),
    re-run ETL to populate DuckDB warehouse, and record sync timestamp.
    """
    settings_path = Path("config/settings.yaml")
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    else:
        settings = {}

    if not seasons:
        # Default to the live season '2627'
        cfg_seasons = settings.get("seasons", ["2627"])
        seasons = [cfg_seasons[-1]] if cfg_seasons else ["2627"]

    fetcher = FootballDataFetcher()
    downloaded = []
    errors = []

    for s in seasons:
        try:
            df, sha = fetcher.fetch_season(s, league=league, force=True)
            downloaded.append({
                "season": s,
                "matches_in_file": len(df),
                "sha": sha[:12],
            })
        except Exception as e:
            errors.append({"season": s, "error": str(e)})

    fixtures_info = None
    if sync_fixtures:
        try:
            f_df, f_sha = fetcher.fetch_fixtures(force=True)
            t1_cnt = len(f_df[f_df["Div"] == league]) if "Div" in f_df.columns else 0
            fixtures_info = {
                "total": len(f_df),
                "league_fixtures": t1_cnt,
                "sha": f_sha[:12],
            }
        except Exception as e:
            fixtures_info = {"error": str(e)}

    # Transform all cached seasons into DuckDB
    etl = WarehouseETL()
    stats = etl.transform_all_cached_seasons()
    etl.close()

    now_utc = datetime.now(timezone.utc)
    sync_result = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_str": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "downloaded_seasons": downloaded,
        "fixtures": fixtures_info,
        "db_matches": stats["matches"],
        "db_odds": stats["odds"],
        "db_seasons": stats["seasons"],
        "errors": errors,
    }

    # Save sync metadata
    p = Path(meta_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sync_result, indent=2, ensure_ascii=False), encoding="utf-8")

    return sync_result


def get_last_sync_info(meta_path: str = "data/sync_meta.json") -> Optional[dict[str, Any]]:
    """Retrieve last synchronization metadata if available."""
    p = Path(meta_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
