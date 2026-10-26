"""
Unit tests for data contracts and automated data warehouse quality audit.
"""

from datetime import datetime, timezone, timedelta
import pytest

from slmodel.data.contracts import TemporalMetadata, MatchDataContract, LineupSnapshotContract, OddsSnapshotContract, OddsSnapshotKind
from slmodel.data.quality_checks import run_data_quality_audit


def test_temporal_metadata_as_of_boundary():
    """Verify that temporal metadata strictly prevents future information leakage."""
    now = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    past = now - timedelta(hours=2)
    future = now + timedelta(hours=2)

    meta_valid = TemporalMetadata(event_time=now, observed_at=past, source="feed")
    assert meta_valid.is_observable_at(now) is True

    meta_future = TemporalMetadata(event_time=now, observed_at=future, source="feed")
    assert meta_future.is_observable_at(now) is False


def test_data_quality_audit_on_warehouse():
    """Verify that the production warehouse passes all integrity and sanity checks."""
    report = run_data_quality_audit("data/warehouse.duckdb")
    assert report.passed is True
    assert len(report.errors) == 0
    assert report.n_checks_run >= 6
    assert report.metrics["total_teams"] >= 18
    assert report.metrics["total_matches"] >= 100
