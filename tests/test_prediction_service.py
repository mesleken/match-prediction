"""
Unit tests for the end-to-end MatchPredictionService and Section 14 report generator.
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from slmodel.services.prediction_service import MatchPredictionService


def test_prediction_service_match_analysis():
    """Verify that MatchPredictionService outputs decoupled models, EV, and section 14 report."""
    service = MatchPredictionService(db_path="data/warehouse.duckdb")
    now_utc = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)

    closing_odds = {"1": 1.65, "X": 3.80, "2": 5.20}

    res = service.analyze_match(
        home_team_id=1,
        away_team_id=2,
        as_of=now_utc,
        closing_odds=closing_odds,
        n_monte_carlo=100,
    )

    # Decoupled probabilities
    assert len(res.p_football_1x2) == 3
    assert np.isclose(sum(res.p_football_1x2), 1.0, atol=1e-2)
    assert res.p_market_1x2 is not None
    assert np.isclose(sum(res.p_market_1x2), 1.0, atol=1e-2)
    assert res.p_hybrid_1x2 is not None
    assert np.isclose(sum(res.p_hybrid_1x2), 1.0, atol=1e-2)

    # xG and Credible Intervals
    assert res.lam_h > 0.0
    assert res.lam_a > 0.0
    assert res.ci_90_lam_h[0] < res.lam_h < res.ci_90_lam_h[1]

    # EV calculations
    assert "1" in res.ev_outcomes
    assert "X" in res.ev_outcomes
    assert "2" in res.ev_outcomes

    # Monte Carlo integration
    assert res.monte_carlo.n_simulations == 100
    assert len(res.top_scores) == 5

    # Markdown report conforms to Section 14
    assert "# Maç Öncesi Analiz" in res.markdown_report
    assert "1. Sonuç Olasılıkları (1X2)" in res.markdown_report
    assert "2. Beklenen Gol ve Skor Dağılımı" in res.markdown_report
    assert "3. Model ve Piyasa Karşılaştırması" in res.markdown_report
    assert "4. Olay Tabanlı Monte Carlo Simülasyon Özeti" in res.markdown_report
    assert "5. Geçmiş Performans & Metodolojik Uyarısı" in res.markdown_report


def test_cli_simulate_command():
    """Verify that CLI simulate command executes cleanly and displays Monte Carlo distributions."""
    from typer.testing import CliRunner
    from slmodel.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["simulate", "Galatasaray", "Fenerbahce", "-n", "50", "-o", "2.20,3.30,3.10", "-t"])
    assert result.exit_code == 0
    assert "MATCHENGINE DİNAMİK MAÇ SİMÜLASYONU" in result.output
    assert "SONUÇ OLASILIKLARI" in result.output
    assert "MONTE CARLO ÇOKLU SENARYO ÇIKTILARI" in result.output
    assert "ÖRNEK SİMÜLE EDİLEN MAÇ AKIŞI" in result.output
