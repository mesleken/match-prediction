"""Unit and integration tests for MatchdayRunner prospective logging and CLV evaluation."""

from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pytest
from typer.testing import CliRunner

from slmodel.cli import app
from slmodel.services.matchday_runner import MatchdayRunner


def test_record_matchday_forecasts(tmp_path):
    """Verify that prospective matchday runner forecasts fixtures and appends to tamper-proof ledger."""
    out_dir = str(tmp_path / "matchday_test")
    runner = MatchdayRunner(db_path="data/warehouse.duckdb")

    # Evaluate season 2627 matchday 1 (which has 8 fixtures in warehouse)
    results = runner.record_matchday_forecasts(
        season="2627",
        matchday=1,
        as_of=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        run_monte_carlo=False,
        n_mc=20,
        output_dir=out_dir,
    )

    assert len(results) > 0
    first = results[0]
    assert "match_id" in first
    assert "pred_id" in first
    assert "row_hash" in first
    assert len(first["row_hash"]) == 64
    assert sum(first["p_football_1x2"]) == pytest.approx(1.0, abs=1e-3)

    # Check disk artifacts
    dossier = Path(first["markdown_path"])
    assert dossier.exists()
    assert "Dixon-Coles" in dossier.read_text(encoding="utf-8")

    index_json = Path(out_dir) / "matchday_forecasts.json"
    assert index_json.exists()


def test_evaluate_matchday_clv(tmp_path):
    """Verify CLV evaluation pipeline correctly extracts edge vs closing odds and realizes RPS."""
    out_dir = str(tmp_path / "clv_test")
    runner = MatchdayRunner(db_path="data/warehouse.duckdb")

    # Ensure matchday 1 has at least one recorded prediction if running standalone
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    cnt = con.execute("""
        SELECT COUNT(*) FROM prediction_log 
        WHERE match_id IN (SELECT match_id FROM matches WHERE season = '2627' AND matchday = 1)
    """).fetchone()[0]
    con.close()

    if cnt == 0:
        runner.record_matchday_forecasts(
            season="2627",
            matchday=1,
            as_of=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            run_monte_carlo=False,
            n_mc=20,
            output_dir=str(tmp_path / "matchday_init"),
        )

    clv_summary = runner.evaluate_matchday_clv(
        season="2627",
        matchday=1,
        output_dir=out_dir,
    )

    assert clv_summary["status"] != "empty"
    assert clv_summary["season"] == "2627"
    assert clv_summary["matchday"] == 1
    assert clv_summary["total_fixtures"] > 0
    assert "positive_clv_rate" in clv_summary
    assert len(clv_summary["fixtures"]) > 0

    report_md = Path(out_dir) / "clv_evaluation_report.md"
    assert report_md.exists()
    assert "Closing Line Value" in report_md.read_text(encoding="utf-8")


def test_cli_matchday_subcommands():
    """Verify Typer CLI integration for `slm matchday record` and `slm matchday evaluate`."""
    runner = CliRunner()

    res_help = runner.invoke(app, ["matchday", "--help"])
    assert res_help.exit_code == 0
    assert "record" in res_help.stdout
    assert "evaluate" in res_help.stdout

    res_rec_help = runner.invoke(app, ["matchday", "record", "--help"])
    assert res_rec_help.exit_code == 0
    assert "--matchday" in res_rec_help.stdout

    res_eval_help = runner.invoke(app, ["matchday", "evaluate", "--help"])
    assert res_eval_help.exit_code == 0
    assert "--matchday" in res_eval_help.stdout
