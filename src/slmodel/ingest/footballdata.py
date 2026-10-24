"""Fetcher and parser for football-data.co.uk CSV files."""

import io
from pathlib import Path
from typing import Any
import pandas as pd
from slmodel.ingest.base import BaseFetcher

class FootballDataFetcher:
    """Manages downloading and inventorying football-data.co.uk datasets."""

    def __init__(self, fetcher: BaseFetcher | None = None) -> None:
        self.fetcher = fetcher or BaseFetcher(cache_dir="data/raw/footballdata")
        self.base_url = "https://www.football-data.co.uk/mmz4281"
        self.fixtures_url = "https://www.football-data.co.uk/fixtures.csv"

    def fetch_season(self, season: str, league: str = "T1", force: bool = False) -> tuple[pd.DataFrame, str]:
        """
        Download and parse a league season CSV (e.g., season='2324', league='T1').
        Returns (DataFrame, sha256_hash).
        """
        url = f"{self.base_url}/{season}/{league}.csv"
        cache_path = f"{league}_{season}.csv"
        data_bytes, sha = self.fetcher.fetch_url(url, cache_subpath=cache_path, force_refresh=force)

        # football-data files may use utf-8-sig (BOM) or windows-1252 encoding
        for encoding in ["utf-8-sig", "windows-1252", "iso-8859-9", "utf-8", "latin-1"]:
            try:
                df = pd.read_csv(io.BytesIO(data_bytes), encoding=encoding)
                # Drop rows where Div or Date is NaN (trailing blank lines)
                if "Div" in df.columns:
                    df = df.dropna(subset=["Div"])
                elif "Date" in df.columns:
                    df = df.dropna(subset=["Date"])
                return df, sha
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode CSV from {url} with standard encodings.")

    def fetch_fixtures(self, force: bool = False) -> tuple[pd.DataFrame, str]:
        """Download upcoming fixtures.csv and parse it."""
        cache_path = "fixtures.csv"
        data_bytes, sha = self.fetcher.fetch_url(self.fixtures_url, cache_subpath=cache_path, force_refresh=force)
        for encoding in ["utf-8-sig", "windows-1252", "utf-8", "latin-1"]:
            try:
                df = pd.read_csv(io.BytesIO(data_bytes), encoding=encoding)
                if "Div" in df.columns:
                    df = df.dropna(subset=["Div"])
                return df, sha
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode fixtures.csv")

    def inspect_season_coverage(self, season: str, league: str = "T1") -> dict[str, Any]:
        """Inspect a season for Phase 0 Data Inventory metrics."""
        try:
            df, sha = self.fetch_season(season, league=league)
        except Exception as e:
            return {
                "season": season,
                "league": league,
                "status": "FAILED_TO_FETCH",
                "error": str(e),
                "row_count": 0,
            }

        cols = list(df.columns)
        closing_odds_cols = [c for c in cols if "C" in c and any(k in c for k in ["PS", "Avg", "Max", "B365"])]
        pinnacle_closing = [c for c in cols if c.startswith("PS") and c.endswith("C") or c in ["PSCH", "PSCD", "PSCA", "PC>2.5", "PC<2.5"]]
        stats_cols = [c for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"] if c in cols]
        xg_cols = [c for c in ["HxG", "AxG"] if c in cols]
        ou_cols = [c for c in cols if ">2.5" in c or "<2.5" in c]
        ah_cols = [c for c in cols if "AH" in c or "Asian" in c]

        teams = set()
        if "HomeTeam" in cols:
            teams.update(df["HomeTeam"].dropna().unique())
        if "AwayTeam" in cols:
            teams.update(df["AwayTeam"].dropna().unique())

        return {
            "season": season,
            "league": league,
            "status": "AVAILABLE",
            "sha256": sha[:12],
            "row_count": len(df),
            "num_teams": len(teams),
            "teams": sorted(list(teams)),
            "match_stats_present": stats_cols,
            "has_full_match_stats": len(stats_cols) >= 8,
            "has_xg": len(xg_cols) == 2,
            "closing_odds_cols": closing_odds_cols,
            "has_pinnacle_closing": len(pinnacle_closing) > 0,
            "has_over_under": len(ou_cols) > 0,
            "has_asian_handicap": len(ah_cols) > 0,
            "date_min": str(df["Date"].dropna().iloc[0]) if "Date" in df.columns and len(df) > 0 else None,
            "date_max": str(df["Date"].dropna().iloc[-1]) if "Date" in df.columns and len(df) > 0 else None,
        }
