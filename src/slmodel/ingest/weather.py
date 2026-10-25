"""Ingestion and matching of historical and forecast stadium weather from Open-Meteo."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd

from slmodel.ingest.http import HttpClient

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherIngestClient:
    """Manages hourly weather collection and alignment for Süper Lig stadiums."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = db_path
        self.http_client = HttpClient(db_path=db_path, min_interval_sec=0.2)

    def fetch_weather_series(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        is_archive: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Fetch hourly weather variables from Open-Meteo for a specific coordinate."""
        url = ARCHIVE_API_URL if is_archive else FORECAST_API_URL
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
            "timezone": "UTC",
        }
        if is_archive:
            params["start_date"] = start_date
            params["end_date"] = end_date
        else:
            params["past_days"] = 3
            params["forecast_days"] = 7

        status, text = self.http_client.get(
            url=url,
            provider="open_meteo",
            params=params,
            ttl_seconds=86400 * 7 if is_archive else 14400,
        )
        if status == 200:
            try:
                return json.loads(text)
            except Exception:
                pass
        return None

    def sync_season_weather(
        self,
        season: str = "2627",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Synchronize weather for all matches in the specified season.
        Aligns each match to the closest hourly observation for the stadium coordinates.
        """
        con = duckdb.connect(self.db_path)
        try:
            # Get matches joined with stadiums
            q = """
                SELECT m.match_id, m.kickoff_utc, m.status, s.stadium_id, s.lat, s.lon, s.name as stadium_name
                FROM matches m
                JOIN stadiums s ON m.stadium_id = s.stadium_id
                WHERE m.season = ? AND s.lat IS NOT NULL AND s.lon IS NOT NULL
                ORDER BY m.kickoff_utc
            """
            matches_df = con.execute(q, [season]).fetchdf()
            if matches_df.empty:
                return {"status": "no_matches", "synced": 0}

            # Group matches by stadium to minimize API queries
            synced_count = 0
            now_utc = datetime.now(timezone.utc)

            for stad_id, g in matches_df.groupby("stadium_id"):
                lat = float(g["lat"].iloc[0])
                lon = float(g["lon"].iloc[0])
                min_dt = pd.to_datetime(g["kickoff_utc"].min())
                max_dt = pd.to_datetime(g["kickoff_utc"].max())

                start_date = min_dt.strftime("%Y-%m-%d")
                end_date = max_dt.strftime("%Y-%m-%d")

                # Fetch archive for historical / played
                weather_data = self.fetch_weather_series(
                    lat=lat,
                    lon=lon,
                    start_date=start_date,
                    end_date=end_date,
                    is_archive=True,
                )

                if not weather_data or "hourly" not in weather_data:
                    # Try forecast API with past_days as fallback for recent days
                    weather_data = self.fetch_weather_series(
                        lat=lat,
                        lon=lon,
                        start_date=start_date,
                        end_date=end_date,
                        is_archive=False,
                    )

                if not weather_data or "hourly" not in weather_data:
                    continue

                hourly = weather_data["hourly"]
                times = pd.to_datetime(hourly["time"], utc=True)
                temps = hourly.get("temperature_2m", [])
                precips = hourly.get("precipitation", [])
                winds = hourly.get("wind_speed_10m", [])
                humidities = hourly.get("relative_humidity_2m", [])

                for _, row in g.iterrows():
                    mid = int(row["match_id"])
                    kickoff = pd.to_datetime(row["kickoff_utc"])
                    if hasattr(kickoff, "tz") and kickoff.tz is None:
                        kickoff = kickoff.tz_localize("UTC")
                    elif hasattr(kickoff, "tz") and kickoff.tz is not None:
                        kickoff = kickoff.tz_convert("UTC")

                    # Find closest hour
                    time_diffs = np.abs((times - kickoff).total_seconds())
                    closest_idx = int(np.argmin(time_diffs))

                    # Only link if within 3 hours
                    if time_diffs[closest_idx] <= 10800:
                        temp = float(temps[closest_idx]) if closest_idx < len(temps) and temps[closest_idx] is not None else 18.0
                        precip = float(precips[closest_idx]) if closest_idx < len(precips) and precips[closest_idx] is not None else 0.0
                        wind = float(winds[closest_idx]) if closest_idx < len(winds) and winds[closest_idx] is not None else 12.0
                        humidity = float(humidities[closest_idx]) if closest_idx < len(humidities) and humidities[closest_idx] is not None else 60.0
                        kind = "archive" if row["status"] == "played" else "forecast"

                        con.execute("""
                            INSERT OR REPLACE INTO match_weather (
                                match_id, temp_c, precip_mm, wind_kmh, humidity_pct, kind, fetched_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [mid, temp, precip, wind, humidity, kind, now_utc])
                        synced_count += 1
        finally:
            con.close()

        return {
            "status": "success",
            "season": season,
            "synced_matches": synced_count,
        }
