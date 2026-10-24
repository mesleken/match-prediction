"""Automatic weekly fixture ingestion and match schedule fetching from Mackolik (arsiv.mackolik.com)."""

from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Optional
import urllib.request

# Canonical Mackolik Team ID to DuckDB Warehouse Team ID mapping
MACKOLIK_TO_WAREHOUSE: dict[int, int] = {
    1: 1,      # Galatasaray
    2: 2,      # Fenerbahçe
    3: 3,      # Beşiktaş
    4: 4,      # Trabzonspor
    451: 5,    # Başakşehir
    447: 6,    # Konyaspor
    656: 8,    # Kasımpaşa
    619: 10,   # Alanyaspor
    572: 12,   # Gaziantep FK
    448: 15,   # Çaykur Rizespor
    15: 17,    # Göztepe
    8: 20,     # Samsunspor
    557: 26,   # Eyüpspor
    7: 27,     # Gençlerbirliği
    9632: 30,  # Erzurumspor FK
    18: 35,    # Kocaelispor
    9665: 38,  # Çorum FK
    591: 39,   # Amedspor
}

WAREHOUSE_NAMES: dict[int, str] = {
    1: "Galatasaray", 2: "Fenerbahce", 3: "Besiktas", 4: "Trabzonspor",
    5: "Basaksehir", 6: "Konyaspor", 8: "Kasimpasa", 10: "Alanyaspor",
    12: "Gaziantep FK", 15: "Rizespor", 17: "Goztepe", 20: "Samsunspor",
    26: "Eyupspor", 27: "Genclerbirligi", 30: "Erzurumspor FK",
    35: "Kocaelispor", 38: "Corum FK", 39: "Amedspor",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://arsiv.mackolik.com/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

DEFAULT_SEASON_ID = 73482  # Mackolik 2026/2027 Süper Lig Season ID


def fetch_fixture_weeks(
    season_id: int = DEFAULT_SEASON_ID,
    cache_path: str = "data/cache/fixture_weeks.json",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch all 34 weeks of the Süper Lig season with dates and completion flags."""
    cache_file = Path(cache_path)
    if not force_refresh and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = f"https://arsiv.mackolik.com/AjaxHandlers/FixtureHandler.aspx?command=getWeeks&id={season_id}"
    req = urllib.request.Request(url, headers=HTTP_HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        weeks_raw = json.loads(raw.replace("'", '"'))
        weeks = []
        for w in weeks_raw:
            weeks.append({
                "week": int(w[0]),
                "start_date": str(w[1]),
                "end_date": str(w[2]),
                "is_finished": bool(w[3] == 1),
            })
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(weeks, ensure_ascii=False, indent=2), encoding="utf-8")
        return weeks
    except Exception as e:
        # Fallback if offline
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        # Generate standard 34-week dummy structure
        return [{"week": i, "start_date": "", "end_date": "", "is_finished": i < 9} for i in range(1, 35)]


def fetch_weekly_fixtures(
    week: int,
    season_id: int = DEFAULT_SEASON_ID,
    cache_dir: str = "data/cache",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch all 9 scheduled/played matches for a specific fixture week from Mackolik."""
    cache_path = Path(cache_dir) / f"fixtures_week_{week}.json"
    if not force_refresh and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = f"https://arsiv.mackolik.com/AjaxHandlers/FixtureHandler.aspx?command=getMatches&id={season_id}&week={week}"
    req = urllib.request.Request(url, headers=HTTP_HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")

        # Clean JavaScript array format into valid JSON
        cleaned = text.replace("'", '"')
        cleaned = re.sub(r",\s*,", ", null,", cleaned)
        cleaned = re.sub(r",\s*,", ", null,", cleaned)
        data = json.loads(cleaned)

        matches = []
        for m in data:
            h_mid = m[3]
            a_mid = m[5]
            h_wid = MACKOLIK_TO_WAREHOUSE.get(h_mid)
            a_wid = MACKOLIK_TO_WAREHOUSE.get(a_mid)

            # Pre-match betting odds from Mackolik
            o1 = float(m[11]) if (len(m) > 11 and m[11] is not None) else 2.10
            ox = float(m[12]) if (len(m) > 12 and m[12] is not None) else 3.30
            o2 = float(m[13]) if (len(m) > 13 and m[13] is not None) else 3.20
            ou = float(m[17]) if (len(m) > 17 and m[17] is not None) else 1.85
            oo = float(m[18]) if (len(m) > 18 and m[18] is not None) else 1.85

            h_name = WAREHOUSE_NAMES.get(h_wid, str(m[4]))
            a_name = WAREHOUSE_NAMES.get(a_wid, str(m[6]))

            status_str = str(m[2]) if len(m) > 2 else "Fikstür"
            matches.append({
                "mackolik_match_id": m[0],
                "week": week,
                "date": str(m[1]),
                "time": str(m[2]),
                "status": status_str,
                "home_id": h_wid,
                "home_name": h_name,
                "away_id": a_wid,
                "away_name": a_name,
                "odds_1": round(o1, 2),
                "odds_x": round(ox, 2),
                "odds_2": round(o2, 2),
                "odds_under": round(ou, 2),
                "odds_over": round(oo, 2),
            })

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
        return matches
    except Exception as e:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Could not fetch fixtures for week {week}: {e}") from e


def get_current_active_week(season_id: int = DEFAULT_SEASON_ID) -> int:
    """Detect the current active/upcoming match week in Süper Lig."""
    weeks = fetch_fixture_weeks(season_id=season_id)
    for w in weeks:
        if not w["is_finished"]:
            return int(w["week"])
    return 9  # Default to week 9 if all finished or unknown
