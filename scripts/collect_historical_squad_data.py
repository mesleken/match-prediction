"""
Collect, synchronize, and report historical player match stats and squad availability coverage.
Ensures zero synthetic data leakage: only authentic season 2627 stats and verified prospective snapshots are maintained.
"""

import sys
import json
from pathlib import Path
from typing import Any
import duckdb

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from slmodel.data.player_performance import generate_and_sync_player_match_stats


def collect_and_sync_all_seasons(
    db_path: str = "data/warehouse.duckdb",
    seasons: list[str] = ("2627",),
    output_json: str = "reports/historical_squad_coverage.json",
) -> dict[str, Any]:
    print("=" * 70)
    print("Squad & Absence Data Synchronization Pipeline")
    print(f"Target authentic seasons: {seasons}")
    print("=" * 70)

    season_results = {}
    total_player_records = 0
    total_matches_covered = 0

    # 1. Sync authentic player_match_stats for current season 2627
    for s in seasons:
        res = generate_and_sync_player_match_stats(db_path=db_path, season=s)
        season_results[s] = res
        total_matches_covered += res["matches_processed"]
        total_player_records += res["records_generated"]
        print(f"Season {s}: {res['matches_processed']} matches, {res['records_generated']} player records synced.")

    con = duckdb.connect(db_path, read_only=True)
    try:
        n_total_snaps = con.execute("SELECT COUNT(*) FROM availability_snapshots").fetchone()[0]
        n_total_pstats = con.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]
    finally:
        con.close()

    summary = {
        "status": "COMPLETED",
        "seasons_covered": list(seasons),
        "total_matches_with_player_stats": total_matches_covered,
        "total_player_match_rows": n_total_pstats,
        "total_availability_snapshots": n_total_snaps,
        "season_breakdown": season_results,
        "historical_seasons_note": "Historical seasons (2122-2526) require external matchday squad archives to avoid synthetic roster lookahead.",
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nCoverage report saved to {output_json}")
    return summary


if __name__ == "__main__":
    collect_and_sync_all_seasons()
