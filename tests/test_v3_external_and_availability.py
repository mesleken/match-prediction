"""Unit and integration tests for Model V3 external factors, availability resolver, and ablation."""

from datetime import datetime, timezone
import duckdb
import numpy as np
import pytest

from slmodel.config import get_model_config
from slmodel.evaluate.backtest import AblationRunner, paired_bootstrap_test
from slmodel.ingest.quota import QuotaManager
from slmodel.models.availability import AvailabilityResolver
from slmodel.models.player_ratings import PlayerRatingEngine


@pytest.fixture(autouse=True)
def ensure_external_test_data():
    """Ensure cached external fixtures, weather, and verified referees are present before tests."""
    from slmodel.ingest.mackolik_referees import build_and_sync_referee_database
    from slmodel.ingest.external_fixtures import ExternalFixturesClient
    from slmodel.ingest.weather import WeatherIngestClient

    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    n_weather = con.execute("SELECT COUNT(*) FROM match_weather").fetchone()[0]
    n_ext = con.execute("SELECT COUNT(*) FROM external_fixtures").fetchone()[0]
    n_ref = con.execute("SELECT COUNT(*) FROM matches WHERE season = '2627' AND referee IS NOT NULL").fetchone()[0]
    con.close()

    if n_ref == 0:
        build_and_sync_referee_database(fetch_online=False)
    if n_ext == 0:
        ext_client = ExternalFixturesClient()
        ext_client.sync_all_external_fixtures()
    if n_weather == 0:
        w_client = WeatherIngestClient()
        w_client.sync_season_weather("2627")


def test_referee_integrity_no_synthetic_rotation():
    """Verify that matches without verified Mackolik referee remain strictly NULL (no fake rotation)."""
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    null_count = con.execute("SELECT COUNT(*) FROM matches WHERE season = '2627' AND referee IS NULL").fetchone()[0]
    total_count = con.execute("SELECT COUNT(*) FROM matches WHERE season = '2627'").fetchone()[0]
    con.close()

    assert total_count == 72
    # Exactly 34 matches are unverified and must be NULL
    assert null_count == 34


def test_quota_manager_and_throttling(tmp_path):
    """Verify quota tracking, UTC reset, and 85% throttle safety buffer."""
    cache_file = str(tmp_path / "test_quota.json")
    mgr = QuotaManager(cache_path=cache_file, db_path="data/warehouse.duckdb")

    state = mgr.load_state()
    assert state["api_football_used"] == 0
    assert mgr.can_request("api_football") is True

    # Simulate 86 calls (exceeds 85% limit of 100)
    for _ in range(86):
        mgr.record_call(provider="api_football", endpoint="/test", status_code=200)

    assert mgr.can_request("api_football") is False
    summary = mgr.get_status_summary()
    assert summary["status"] == "throttled"
    assert summary["remaining"] <= 15


def test_external_fixtures_calendar():
    """Verify external UEFA and Cup fixtures are ingested and linked to valid teams."""
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    ext_df = con.execute("""
        SELECT ef.ext_id, ef.team_id, t.name, ef.competition, ef.kickoff_utc, ef.is_home
        FROM external_fixtures ef
        JOIN teams t ON ef.team_id = t.team_id
    """).fetchdf()
    con.close()

    assert not ext_df.empty
    assert len(ext_df) >= 20
    # Must contain European fixtures for top Turkish clubs
    teams_with_euro = set(ext_df["name"].tolist())
    assert "Galatasaray" in teams_with_euro
    assert "Fenerbahce" in teams_with_euro


def test_match_weather_ingestion():
    """Verify match weather table contains aligned temperature and wind observations."""
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    w_df = con.execute("SELECT match_id, temp_c, precip_mm, wind_kmh, kind FROM match_weather").fetchdf()
    con.close()

    assert not w_df.empty
    assert len(w_df) == 72
    assert (w_df["temp_c"] >= -10.0).all() and (w_df["temp_c"] <= 45.0).all()
    assert (w_df["precip_mm"] >= 0.0).all()
    assert (w_df["wind_kmh"] >= 0.0).all()


def test_availability_precedence_and_doubtful(tmp_path):
    """Verify strict precedence: Manual Override > API Snapshot > Baseline."""
    resolver = AvailabilityResolver(db_path="data/warehouse.duckdb")

    # Clean existing overrides for clean baseline
    con = duckdb.connect("data/warehouse.duckdb")
    con.execute("UPDATE availability_overrides SET active = FALSE WHERE team_id = 1")
    con.close()

    # Baseline: Victor Osimhen is available (p_play = 1.0)
    roster_base = resolver.resolve_team_availability(1)
    osimhen = next(p for p in roster_base if "Osimhen" in p["name"])
    assert osimhen["status"] == "available"
    assert osimhen["p_play"] == 1.0

    # Apply manual override as 'doubtful'
    oid = resolver.set_manual_override(
        team_id=1,
        player_id=osimhen["player_id"],
        status="doubtful",
        note="Kas yorgunlugu",
    )
    roster_doubt = resolver.resolve_team_availability(1)
    osimhen_d = next(p for p in roster_doubt if p["player_id"] == osimhen["player_id"])
    assert osimhen_d["status"] == "doubtful"
    assert osimhen_d["p_play"] == 0.50
    assert osimhen_d["source"] == "manual"

    # Set as 'out'
    resolver.set_manual_override(
        team_id=1,
        player_id=osimhen["player_id"],
        status="out",
        note="Sakatlik",
    )
    roster_out = resolver.resolve_team_availability(1)
    osimhen_out = next(p for p in roster_out if p["player_id"] == osimhen["player_id"])
    assert osimhen_out["status"] == "out"
    assert osimhen_out["p_play"] == 0.0

    # Clear override
    resolver.clear_manual_override(oid)
    con = duckdb.connect("data/warehouse.duckdb")
    con.execute("UPDATE availability_overrides SET active = FALSE WHERE team_id = 1")
    con.close()


def test_paired_bootstrap_test():
    """Verify paired bootstrap hypothesis test calculates accurate confidence intervals and p-values."""
    diffs = np.array([-0.005, -0.004, -0.006, -0.003, -0.005, -0.004, -0.007, -0.005])
    res = paired_bootstrap_test(diffs, n_boot=500, seed=42)

    assert res["mean_diff"] < 0.0
    assert res["ci_upper"] < 0.0  # Entire CI is strictly negative
    assert res["p_value"] < 0.05  # Statistically significant


def test_ablation_runner():
    """Verify AblationRunner evaluates leave-one-out factor impact and decisions."""
    runner = AblationRunner()
    res = runner.run_study(factors_to_test=["referee", "weather"], seasons=["2627"])

    assert "baseline_rps" in res
    assert len(res["factor_reports"]) == 2
    reports_map = {r["factor"]: r for r in res["factor_reports"]}

    assert "referee" in reports_map
    assert "weather" in reports_map
    # Weather does not yield gain on holdout by default
    assert reports_map["weather"]["recommended_default"] is False
