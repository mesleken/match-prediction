"""Command line interface for Süper Lig Model (slm)."""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

# Ensure robust UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml
import typer
import duckdb
import numpy as np
import pandas as pd
import rapidfuzz.process
from slmodel.ingest.footballdata import FootballDataFetcher
from slmodel.ingest.sync import sync_latest_data, get_last_sync_info
from slmodel.etl.transform import WarehouseETL
from slmodel.evaluate.backtest import WalkForwardBacktest, AblationRunner
from slmodel.evaluate.season_stats import compute_season_statistics, format_season_label
from slmodel.ingest.external_fixtures import ExternalFixturesClient
from slmodel.ingest.weather import WeatherIngestClient
from slmodel.ingest.api_football import ApiFootballClient
from slmodel.models.availability import AvailabilityResolver
from slmodel.models.dixon_coles import DixonColesModel
from slmodel.models.player_ratings import PlayerRatingEngine
from slmodel.models.market import devig_shin, invert_lambdas, blend_lambdas
from slmodel.models.htft import predict_ht_ft, predict_half_markets
from slmodel.models.referee_engine import RefereeEngine
from slmodel.models.sidemodels import SideMarketsModel, predict_corners, predict_cards
from slmodel.features.context import ContextFeatureExtractor
from slmodel.data.player_performance import (
    generate_and_sync_player_match_stats,
    compute_player_season_stats_and_ratings,
    get_player_season_leaderboards,
)
from slmodel.simulate.score_matrix import score_matrix
from slmodel.simulate.markets import (
    one_x_two,
    draw_no_bet,
    double_chance,
    over_under,
    both_teams_to_score,
    goal_range_probabilities,
    clean_sheet_and_win_to_nil,
    asian_handicap,
    most_likely_scores,
    kelly_stake,
)
from slmodel.simulate.bootstrap import (
    bayesian_bootstrap_prediction,
    laplace_posterior_prediction,
)
from slmodel.tracking.prediction_log import PredictionLogger, verify_prediction_log

app = typer.Typer(
    name="slm",
    help="Süper Lig Match Prediction Engine CLI",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="info")
def info() -> None:
    """Print system and model engine status."""
    typer.echo("Süper Lig Model Engine (slm) v0.2.0 (Composite MAP + Laplace Posterior Predictive)")
    typer.echo("Directory: " + str(Path.cwd()))


@app.command(name="inventory")
def inventory() -> None:
    """Run Phase 0 Data Inventory checks on football-data.co.uk and generate report."""
    settings_path = Path("config/settings.yaml")
    if not settings_path.exists():
        typer.secho("Error: config/settings.yaml not found", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    seasons = settings.get("seasons", [])
    league = settings.get("league", "T1")

    typer.echo(f"Starting Phase 0 Data Inventory for {league} across {len(seasons)} seasons...")
    fetcher = FootballDataFetcher()

    results = []
    for s in seasons:
        typer.echo(f"  Checking season {s}...")
        cov = fetcher.inspect_season_coverage(s, league=league)
        results.append(cov)

    fixtures_cov = {}
    try:
        f_df, _ = fetcher.fetch_fixtures()
        t1_fixtures = f_df[f_df["Div"] == league] if "Div" in f_df.columns else f_df
        fixtures_cov = {
            "status": "AVAILABLE",
            "total_fixtures": len(f_df),
            "t1_fixtures_count": len(t1_fixtures),
            "cols": list(f_df.columns),
            "has_odds": any(c in f_df.columns for c in ["B365H", "PSH", "AvgH", "MaxH"]),
        }
    except Exception as e:
        fixtures_cov = {"status": "UNAVAILABLE", "error": str(e)}

    report_lines = [
        "# Faz 0 — Veri Envanteri Raporu (Data Inventory Report)",
        "",
        "Bu rapor, `football-data.co.uk` üzerinden çekilen ham Süper Lig (T1) verilerini, sütun doluluklarını, kapanış oranlarını ve maç istatistiklerini belgeler.",
        "",
        "## 1. Süper Lig (T1) Sezon Kapsamı ve Doluluk Tablosu",
        "",
        "| Sezon | Durum | Maç Sayısı | Takım Sayısı | Şut/Korner/Kart? | xG (HxG/AxG)? | Kapanış Oranları? | Pinnacle Kapanış? | Alt/Üst? | Asya Handikap? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    total_matches = 0
    available_seasons = 0

    for r in results:
        if r["status"] == "AVAILABLE":
            available_seasons += 1
            total_matches += r["row_count"]
            stats_str = "EVET" if r["has_full_match_stats"] else f"Kısmi ({len(r['match_stats_present'])})"
            xg_str = "EVET" if r.get("has_xg") else "YOK"
            close_str = "EVET" if len(r["closing_odds_cols"]) > 0 else "YOK"
            ps_str = "EVET" if r["has_pinnacle_closing"] else "YOK"
            ou_str = "EVET" if r["has_over_under"] else "YOK"
            ah_str = "EVET" if r["has_asian_handicap"] else "YOK"
            report_lines.append(
                f"| **{r['season']}** | {r['status']} | {r['row_count']} | {r['num_teams']} | {stats_str} | {xg_str} | {close_str} | {ps_str} | {ou_str} | {ah_str} |"
            )
        else:
            report_lines.append(f"| **{r['season']}** | {r['status']} | 0 | 0 | - | - | - | - | - | - |")

    report_lines.extend([
        "",
        f"**Toplam Kullanılabilir Sezon:** {available_seasons} / {len(seasons)}  ",
        f"**Toplam Maç Sayısı:** {total_matches}  ",
        "",
        "## 2. Yaklaşan Maç Fikstürü (`fixtures.csv`) Kontrolü (G3 Kapısı)",
        "",
    ])

    if fixtures_cov.get("status") == "AVAILABLE":
        report_lines.extend([
            f"- **Durum:** Erişilebilir (Toplam {fixtures_cov['total_fixtures']} maç)",
            f"- **Süper Lig ({league}) Maç Sayısı:** {fixtures_cov['t1_fixtures_count']} maç",
            f"- **Bahis Oranları Mevcut mu:** {'EVET' if fixtures_cov['has_odds'] else 'HAYIR'}",
        ])
    else:
        report_lines.extend([
            f"- **Durum:** Erişilemedi / Yok ({fixtures_cov.get('error')})",
        ])

    out_path = Path("reports/data_inventory.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    typer.secho(f"\nPhase 0 Data Inventory written to {out_path}", fg=typer.colors.GREEN, bold=True)


@app.command(name="sync")
def sync_command(
    season: str = typer.Option("2627", "--season", "-s", help="Season code to fetch fresh from web (e.g. 2627, or 'all')"),
    fixtures: bool = typer.Option(True, "--fixtures/--no-fixtures", help="Also fetch upcoming fixtures.csv"),
) -> None:
    """Download freshest match data, xG, and odds from web and refresh DuckDB warehouse."""
    typer.echo(f"Starting live data synchronization (Season: {season}, fixtures: {fixtures})...")
    seasons_to_fetch = None if season == "all" else [season]
    res = sync_latest_data(seasons=seasons_to_fetch, sync_fixtures=fixtures)

    if res.get("errors"):
        typer.secho(f"Encountered warnings/errors during download: {res['errors']}", fg=typer.colors.YELLOW)

    typer.secho("\n" + "=" * 64, fg=typer.colors.CYAN)
    typer.secho("  SYNCHRONIZATION COMPLETE (DuckDB Warehouse Refreshed)", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho("=" * 64, fg=typer.colors.CYAN)
    typer.echo(f"  Time (UTC):       {res['timestamp_str']}")
    for d in res["downloaded_seasons"]:
        typer.echo(f"  Season {d['season']}:       {d['matches_in_file']} matches downloaded (Hash: {d['sha']})")
    if res.get("fixtures"):
        f_info = res["fixtures"]
        if "error" in f_info:
            typer.echo(f"  Fixtures:         Failed ({f_info['error']})")
        else:
            typer.echo(f"  Fixtures:         {f_info['total']} total ({f_info['league_fixtures']} Süper Lig matches)")
    typer.echo(f"  Warehouse Total:  {res['db_matches']} matches across {res['db_seasons']} seasons, {res['db_odds']} odds snapshots.")
    typer.secho("=" * 64 + "\n", fg=typer.colors.CYAN)


@app.command(name="ingest-mackolik")
def ingest_mackolik_command(
    db_path: str = typer.Option("data/warehouse.duckdb", "--db-path", help="Path to DuckDB warehouse"),
) -> None:
    """Fetch authentic 2026/27 squads, statistics, and ratings from Mackolik (arsiv.mackolik.com)."""
    from slmodel.ingest.mackolik_squads import fetch_and_sync_all_mackolik_squads

    typer.secho("Fetching authentic 2026/27 squads from arsiv.mackolik.com...", fg=typer.colors.CYAN)
    res = fetch_and_sync_all_mackolik_squads(db_path=db_path)
    typer.secho(f"\n[OK] Mackolik Senkronizasyonu Tamamlandı! {res['players_synced']} oyuncu ({res['teams_synced']} takım) DuckDB'ye aktarıldı.", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  İşlenen maç sayısı: {res['matches_processed']}")
    typer.echo(f"  Bayesyen reyting güncellenen oyuncu sayısı: {res['season_records']}")
    if res.get("errors"):
        typer.secho(f"  Uyarılar: {res['errors']}", fg=typer.colors.YELLOW)


@app.command(name="fixtures")
def fixtures_command(
    week: Optional[int] = typer.Option(None, "--week", "-w", help="Week number (1-34). Defaults to active week."),
) -> None:
    """Display weekly Süper Lig match fixtures with real bookmaker odds from Mackolik."""
    from slmodel.ingest.fixtures import fetch_weekly_fixtures, get_current_active_week

    active_week = get_current_active_week()
    target_week = week if week is not None else active_week

    typer.echo(f"Fetching fixtures for Süper Lig Week {target_week} from Mackolik...")
    matches = fetch_weekly_fixtures(target_week)

    typer.secho("\n" + "=" * 84, fg=typer.colors.CYAN)
    typer.secho(f"  SÜPER LİG 2026/27 — {target_week}. HAFTA FİKSTÜRÜ VE ORANLARI", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho("=" * 84, fg=typer.colors.CYAN)
    typer.echo(f"  {'Tarih/Saat':<14} | {'Ev Sahibi':<16} vs {'Deplasman':<16} | {'1':>5} {'X':>5} {'2':>5} | {'Üst 2.5':>7} {'Alt 2.5':>7}")
    typer.echo("-" * 84)
    for m in matches:
        dt_str = f"{m['date']} {m['time']}"
        typer.echo(f"  {dt_str:<14} | {m['home_name']:<16} vs {m['away_name']:<16} | {m['odds_1']:>5.2f} {m['odds_x']:>5.2f} {m['odds_2']:>5.2f} | {m['odds_over']:>7.2f} {m['odds_under']:>7.2f}")
    typer.secho("=" * 84 + "\n", fg=typer.colors.CYAN)


@app.command(name="etl")
def run_etl() -> None:
    """Run ETL to parse all cached CSV files into DuckDB warehouse."""
    typer.echo("Starting ETL transformation into DuckDB warehouse (data/warehouse.duckdb)...")
    etl = WarehouseETL()
    stats = etl.transform_all_cached_seasons()
    etl.close()
    typer.secho(
        f"ETL Complete! Ingested {stats['seasons']} seasons, {stats['matches']} matches, {stats['odds']} odds snapshots.",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command(name="backtest")
def run_backtest(
    from_season: str = typer.Option("1819", help="Start evaluation season (e.g. 1819)"),
    to_season: str = typer.Option("2627", help="End evaluation season (e.g. 2627)"),
    refit_interval: int = typer.Option(10, help="Matches between model refits"),
) -> None:
    """Run time-ordered walk-forward backtest and print benchmark comparison."""
    typer.echo(f"Running walk-forward backtest (Seasons {from_season}..{to_season}, refit every {refit_interval} matches)...")
    bt = WalkForwardBacktest(
        refit_interval=refit_interval,
        burn_in_matches=306,
    )
    summary = bt.run(from_season=from_season, to_season=to_season)

    typer.echo("\n" + "=" * 68)
    typer.secho("WALK-FORWARD BACKTEST RESULTS SUMMARY (OUT-OF-SAMPLE)", fg=typer.colors.CYAN, bold=True)
    typer.echo("=" * 68)
    typer.echo(f"Evaluated Matches:          {summary.get('evaluated_matches')}")
    typer.echo(f"Naive Baseline RPS:         {summary.get('mean_rps_naive'):.4f}")
    typer.echo(f"Dixon-Coles Hybrid RPS:     {summary.get('mean_rps_dc'):.4f} (diff vs naive: {summary.get('dc_vs_naive_rps_diff'):+.4f})")
    typer.echo(f"Dixon-Coles Score Log-Loss: {summary.get('mean_logloss_dc'):.4f}")
    typer.echo(f"Dixon-Coles Brier Score:    {summary.get('mean_brier_dc'):.4f}")
    typer.echo(f"Home Win Calibration ECE:   {summary.get('ece_home_win'):.4f}")

    if "market_matches_count" in summary:
        typer.echo("-" * 68)
        typer.secho(f"MARKET COMPARISON SUBSET ({summary.get('market_matches_count')} matches with closing odds):", fg=typer.colors.YELLOW, bold=True)
        typer.echo(f"  Naive Baseline RPS:       {summary.get('subset_rps_naive'):.4f}")
        typer.echo(f"  Dixon-Coles Hybrid RPS:   {summary.get('subset_rps_dc'):.4f}")
        typer.echo(f"  Market Closing Devig RPS: {summary.get('subset_rps_market'):.4f}")
        typer.echo(f"  Blended (DC + Market) RPS:{summary.get('subset_rps_blend'):.4f}")
        typer.echo(f"  Blended Score Log-Loss:   {summary.get('subset_logloss_blend'):.4f}")
        typer.echo(f"  Paired Difference SD (s_d):{summary.get('paired_diff_sd'):.4f}")
        typer.echo(f"  Min Detectable Effect MDE: {summary.get('min_detectable_effect_mde'):.4f} (80% power at alpha=0.05)")
        typer.echo(f"  Blend vs Market diff:     {summary.get('blend_vs_market_diff'):+.5f}")
    typer.echo("=" * 68 + "\n")

    report_lines = [
        "# Faz 1 & 2 — Geliştirilmiş Matematiksel Model ve Walk-Forward Backtest Raporu",
        "",
        f"**Değerlendirilen Sezonlar:** {from_season} – {to_season} (2026/27 canlı sezon dahil)  ",
        f"**Toplam Maç Sayısı:** {summary.get('evaluated_matches')}  ",
        "",
        "## Referans Merdiveni (Benchmark Comparison)",
        "",
        "| Model Seviyesi | RPS (Düşük = İyi) | Fark vs Naif | Skor Log-Kaybı | Not |",
        "|---|---|---|---|---|",
        f"| **B0: Naif Baz (Tarihsel Frekans)** | {summary.get('mean_rps_naive'):.4f} | 0.0000 | - | Geçmiş maç dağılımı |",
        f"| **B3: Geliştirilmiş Dixon-Coles (Katman A+E)** | {summary.get('mean_rps_dc'):.4f} | {summary.get('dc_vs_naive_rps_diff'):+.4f} | {summary.get('mean_logloss_dc'):.4f} | Şut & xG Hibrit + Laplace + EB Önsel |",
    ]

    if "market_matches_count" in summary:
        report_lines.extend([
            f"| **B4: Piyasa Kapanış (Shin Devig)** | {summary.get('subset_rps_market'):.4f} | - | - | Bahis bürosu kapanış konsensüsü |",
            f"| **Katman F: Çoklu-Piyasa Harmanı (1X2 + O/U)** | {summary.get('subset_rps_blend'):.4f} | - | {summary.get('subset_logloss_blend'):.4f} | İki ağırlıklı zımni harman (u, v) |",
            "",
            "## İstatistiksel Güç ve Kalibrasyon Metrikleri",
            "",
            f"- **Kapanış Oranlı Maç Sayısı (N):** {summary.get('market_matches_count')}",
            f"- **Brier Skoru (DC):** {summary.get('mean_brier_dc'):.4f}",
            f"- **Beklenen Kalibrasyon Hatası (ECE):** {summary.get('ece_home_win'):.4f}",
            f"- **Eşleştirilmiş Fark Standart Sapması ($s_d$):** {summary.get('paired_diff_sd'):.4f}",
            f"- **Saptanabilir Minimum Etki (MDE, %80 güç):** {summary.get('min_detectable_effect_mde'):.4f}",
            f"- **Harman vs Piyasa Farkı:** {summary.get('blend_vs_market_diff'):+.5f}",
        ])

    if "season_breakdown" in summary:
        report_lines.extend([
            "",
            "## Sezon Bazlı Performans Dağılımı (2018/19 – 2026/27)",
            "",
            "| Sezon | Maç Sayısı | Naif RPS | Saf Model RPS | Saf Skor Log-Loss | Piyasa Kapanış RPS | Harmanlanmış RPS | Harman Skor Log-Loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for sr in summary["season_breakdown"]:
            s_lbl = format_season_label(sr["season"])
            r_mkt_s = f"{sr['rps_market']:.4f}" if sr["rps_market"] is not None else "-"
            r_bld_s = f"{sr['rps_blend']:.4f}" if sr["rps_blend"] is not None else "-"
            ll_bld_s = f"{sr['logloss_blend']:.4f}" if sr["logloss_blend"] is not None else "-"
            report_lines.append(
                f"| **{s_lbl} (`{sr['season']}`)** | {sr['matches']} | {sr['rps_naive']:.4f} | **{sr['rps_dc']:.4f}** | {sr['logloss_dc']:.4f} | {r_mkt_s} | **{r_bld_s}** | {ll_bld_s} |"
            )

    out_p = Path("reports/backtest_baseline.md")
    out_p.write_text("\n".join(report_lines), encoding="utf-8")
    typer.secho(f"Backtest report saved to {out_p}", fg=typer.colors.GREEN)


@app.command(name="stats")
def stats_command(
    season: str = typer.Option("2627", "--season", "-s", help="Season code to inspect (default: 2627 for 2026/27)"),
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB database"),
) -> None:
    """Display and export comprehensive season statistics, standings, xG/xPts, shots, corners, and cards."""
    summary = compute_season_statistics(db_path=db_path, season=season)
    if summary["matches_played"] == 0:
        typer.secho(f"No played matches found for season {season}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    s_label = summary["season_label"]
    df_st = summary["standings"]

    typer.secho("\n" + "=" * 96, fg=typer.colors.CYAN, bold=True)
    typer.secho(
        f"  SÜPER LİG {s_label} SEZONU İSTATİSTİKLERİ, PUAN DURUMU VE xG / xPts ANALİZİ",
        fg=typer.colors.BRIGHT_GREEN,
        bold=True,
    )
    typer.secho("=" * 96, fg=typer.colors.CYAN)
    typer.echo(
        f"  Oynanan Maç: {summary['matches_played']} ({summary['date_min']} .. {summary['date_max']}) | "
        f"Takım: {summary['teams_count']} | Maç Başına Gol: {summary['goals_per_match']:.2f} (İY: {summary['ht_goals_per_match']:.2f})"
    )
    typer.echo(
        f"  Ev: %{summary['home_win_pct']:.1f} | Beraberlik: %{summary['draw_pct']:.1f} | Deplasman: %{summary['away_win_pct']:.1f} | "
        f"Üst 2.5: %{summary['over25_pct']:.1f} | KG Var: %{summary['btts_pct']:.1f}"
    )
    typer.echo(
        f"  Maç Başına xG: {summary['xg_per_match']:.2f} | Şut/M: {summary['shots_per_match']:.1f} "
        f"(İsabetli: {summary['sot_per_match']:.1f}) | Korner/M: {summary['corners_per_match']:.2f} | "
        f"Sarı Kart/M: {summary['yellow_per_match']:.2f} | Kırmızı/M: {summary['red_per_match']:.2f}"
    )
    typer.echo("-" * 96)

    display_cols = [
        "Sıra", "Takım", "OM", "G", "B", "M", "AG", "YG", "AV", "Puan",
        "xPts", "xG", "xGA", "xGD", "Şut/M", "İsabetli Şut/M", "Korner/M", "Sarı Kart/M", "Kırmızı Kart",
    ]
    fmt_df = df_st[display_cols].copy()
    for c in ["xPts", "xG", "xGA", "xGD", "Şut/M", "İsabetli Şut/M", "Korner/M", "Sarı Kart/M"]:
        fmt_df[c] = fmt_df[c].map(lambda v: f"{v:.2f}")
    typer.echo(fmt_df.to_string(index=False))
    typer.secho("=" * 96 + "\n", fg=typer.colors.CYAN)

    # Export markdown report
    md_lines = [
        f"# Süper Lig {s_label} (`{season}`) Sezon İstatistikleri ve xG Analiz Raporu",
        "",
        f"- **Kapsam:** {summary['matches_played']} maç ({summary['date_min']} – {summary['date_max']}), {summary['teams_count']} takım",
        f"- **Gol Ortalaması:** Maç başına **{summary['goals_per_match']:.2f}** gol (İlk yarı: **{summary['ht_goals_per_match']:.2f}**), Maç başına xG: **{summary['xg_per_match']:.2f}**",
        f"- **Sonuç Dağılımı:** Ev Sahibi **%{summary['home_win_pct']:.1f}** | Beraberlik **%{summary['draw_pct']:.1f}** | Deplasman **%{summary['away_win_pct']:.1f}**",
        f"- **Alt/Üst & KG:** 2.5 Üst **%{summary['over25_pct']:.1f}** | KG Var **%{summary['btts_pct']:.1f}**",
        f"- **Yan Piyasalar:** Maç başına **{summary['shots_per_match']:.1f}** şut (**{summary['sot_per_match']:.1f}** isabetli), **{summary['corners_per_match']:.2f}** korner, **{summary['fouls_per_match']:.1f}** faul, **{summary['yellow_per_match']:.2f}** sarı kart, **{summary['red_per_match']:.2f}** kırmızı kart",
        "",
        "## 1. Puan Durumu, Beklenen Puan (xPts), xG ve Takım İstatistikleri",
        "",
        "| Sıra | Takım | OM | G | B | M | AG | YG | AV | Puan | xPts | P-xPts | xG | xGA | xGD | G-xG | Şut/M | İ.Şut/M | Korner/M | Y.Kor/M | Sarı/M | Kırmızı | Üst 2.5% | KG Var% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_st.iterrows():
        md_lines.append(
            f"| {int(r['Sıra'])} | **{r['Takım']}** | {int(r['OM'])} | {int(r['G'])} | {int(r['B'])} | {int(r['M'])} | "
            f"{int(r['AG'])} | {int(r['YG'])} | {int(r['AV']):+d} | **{int(r['Puan'])}** | {r['xPts']:.1f} | {r['Puan - xPts']:+.1f} | "
            f"{r['xG']:.2f} | {r['xGA']:.2f} | {r['xGD']:+.2f} | {r['G - xG']:+.2f} | {r['Şut/M']:.1f} | {r['İsabetli Şut/M']:.1f} | "
            f"{r['Korner/M']:.1f} | {r['Y.Korner/M']:.1f} | {r['Sarı Kart/M']:.2f} | {int(r['Kırmızı Kart'])} | "
            f"%{r['Üst 2.5 %']:.0f} | %{r['KG Var %']:.0f} |"
        )

    out_path = Path(f"reports/season_{season}_stats.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    typer.secho(f"Season statistics report saved to {out_path}", fg=typer.colors.GREEN)


def _resolve_team(query: str, con: duckdb.DuckDBPyConnection) -> tuple[int, str]:
    teams = con.execute("SELECT team_id, name FROM teams").fetchall()
    aliases = con.execute("SELECT team_id, source_name FROM team_aliases").fetchall()
    mapping: dict[str, tuple[int, str]] = {}
    for tid, name in teams:
        mapping[name.lower()] = (tid, name)
    for tid, sname in aliases:
        cname = next((n for t, n in teams if t == tid), sname)
        mapping[sname.lower()] = (tid, cname)

    q_lower = query.strip().lower()
    if q_lower in mapping:
        return mapping[q_lower]

    match_res = rapidfuzz.process.extractOne(q_lower, list(mapping.keys()))
    if match_res and match_res[1] >= 65:
        return mapping[match_res[0]]

    raise ValueError(f"Team '{query}' not recognized in Süper Lig database.")


@app.command(name="lineup")
def lineup_command(
    team: str = typer.Argument(..., help="Team name or abbreviation (e.g. Galatasaray, Fenerbahce)"),
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB database"),
) -> None:
    """Display 2026/27 team squad, ratings, tactical unit synergies, and marginal player importance."""
    con = duckdb.connect(db_path)
    try:
        t_id, t_name = _resolve_team(team, con)
    finally:
        con.close()

    eng = PlayerRatingEngine(db_path=db_path)
    starters = eng.get_default_lineup(t_id)
    roster = eng.get_team_roster(t_id)
    synergies = eng._cache_synergies.get(t_id, [])
    importance = eng.evaluate_player_importance(t_id)

    typer.secho("\n" + "=" * 84, fg=typer.colors.CYAN, bold=True)
    typer.secho(f"  {t_name.upper()} — 2026/27 OYUNCU KADROSU, REYTİNGLER VE TANDEM SİNERJİLERİ", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho("=" * 84, fg=typer.colors.CYAN)

    typer.secho("\n[VARSAYILAN İLK 11 (BENCHMARK BAŞLANGIÇ KADROSU)]", fg=typer.colors.YELLOW, bold=True)
    typer.echo(" Mevki | Oyuncu                   | Yaş | Hücum (att) | Savunma (def) | Genel (OVR) | Değer (€)")
    typer.echo("-" * 84)
    for p in starters:
        typer.echo(
            f"  {p['position']:4s} | {p['name']:24s} | {p['age']:3d} | {p['att_rating']:+11.2f} | {p['def_rating']:+13.2f} | {p['overall_rating']:11.0f} | €{p['market_value_eur']:,}"
        )

    bench = [p for p in roster if not p.get("is_starter")]
    if bench:
        typer.secho("\n[YEDEK KADRO (ROTASYON VE YEDEK OPSİYONLARI)]", fg=typer.colors.YELLOW, bold=True)
        for p in bench:
            typer.echo(
                f"  {p['position']:4s} | {p['name']:24s} | {p['age']:3d} | {p['att_rating']:+11.2f} | {p['def_rating']:+13.2f} | {p['overall_rating']:11.0f} | €{p['market_value_eur']:,}"
            )

    if synergies:
        typer.secho("\n[TAKTIKSEL TANDEM SİNERJİLERİ (KATMAN C - COHESION)]", fg=typer.colors.YELLOW, bold=True)
        id_to_name = {p["player_id"]: p["name"] for p in roster}
        for s in synergies:
            p1 = id_to_name.get(s["player1_id"], f"P{s['player1_id']}")
            p2 = id_to_name.get(s["player2_id"], f"P{s['player2_id']}")
            typer.echo(f"  {s['unit_type'].capitalize():8s} Tandemi: {p1} <--> {p2} (Sinerji Çarpanı: x{s['synergy_score']:.2f})")

    if importance:
        typer.secho("\n[EN VAZGEÇİLMEZ OYUNCULAR (MARJİNAL YOKLUK ETKİSİ)]", fg=typer.colors.YELLOW, bold=True)
        typer.echo("  Oyuncu                   | Mevki | Yedek İkame              | Galibiyet Kaybı | Gol Averajı Kaybı")
        typer.echo("-" * 84)
        for imp in importance[:6]:
            typer.echo(
                f"  {imp['name']:24s} | {imp['position']:5s} | {imp['backup_name']:24s} | {imp['marginal_drop_win_pct']:+13.1f}% | {imp['marginal_goal_diff_loss']:+14.2f} gol"
            )
    typer.secho("=" * 84 + "\n", fg=typer.colors.CYAN)


@app.command(name="predict")
def predict(
    home: str = typer.Argument(..., help="Home team name or abbreviation (e.g. Galatasaray, GS)"),
    away: str = typer.Argument(..., help="Away team name or abbreviation (e.g. Fenerbahce, FB)"),
    odds: Optional[str] = typer.Option(None, "--odds", "-o", help="Pre-match 1X2 odds separated by commas (e.g. '2.10,3.40,3.60')"),
    ou25_odds: Optional[str] = typer.Option(None, "--ou25", help="Pre-match Over/Under 2.5 odds separated by comma (e.g. '1.75,2.05')"),
    home_missing: Optional[str] = typer.Option(None, "--home-missing", "-hm", help="Comma-separated missing players for home team (e.g. 'Osimhen, Torreira')"),
    away_missing: Optional[str] = typer.Option(None, "--away-missing", "-am", help="Comma-separated missing players for away team (e.g. 'Fred, Dzeko')"),
    referee: Optional[str] = typer.Option(None, "--referee", "-r", help="Match referee name (e.g. 'Halil Umut Meler', 'Ali Sansalan')"),
    bootstrap: bool = typer.Option(False, "--bootstrap", "-b", help="Run full non-parametric Dirichlet Bayesian bootstrap (50 draws)"),
    log: bool = typer.Option(False, "--log", "-l", help="Record prediction to tamper-proof cryptographic ledger"),
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB database"),
) -> None:
    """Generate comprehensive probabilistic match prediction for a Süper Lig fixture."""
    con = duckdb.connect(db_path)
    try:
        h_id, h_name = _resolve_team(home, con)
        a_id, a_name = _resolve_team(away, con)
        matches_df = con.execute(
            """
            SELECT 
                m.match_id, m.season, m.kickoff_utc, m.home_team_id, m.away_team_id,
                m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
                m.status, m.crowd, m.is_derby,
                sh.shots as hs, sh.shots_on_target as hst, sh.corners as hc,
                sh.fouls as hf, sh.yellow as hy, sh.red as hr, sh.xg as hxg,
                sa.shots as as_, sa.shots_on_target as ast, sa.corners as ac,
                sa.fouls as af, sa.yellow as ay, sa.red as ar, sa.xg as axg
            FROM matches m
            LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
            LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
            WHERE m.status = 'played' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
            ORDER BY m.kickoff_utc, m.match_id
            """
        ).fetchdf()
    finally:
        con.close()

    typer.secho("\n" + "=" * 72, fg=typer.colors.CYAN, bold=True)
    typer.secho(f"  MATCH PREDICTION: {h_name} vs {a_name}", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho("=" * 72, fg=typer.colors.CYAN)

    now_utc = datetime.now(timezone.utc)

    # 1. Referee & Context Features (Katman E & Katman G)
    ref_engine = RefereeEngine(db_path=db_path)
    ref_prof = None
    if referee:
        ref_prof = ref_engine.get_referee_profile(referee, as_of=now_utc, matches_history=matches_df)

    ctx_extractor = ContextFeatureExtractor(db_path=db_path)
    is_derby = ctx_extractor.is_derby(h_id, a_id)
    ctx_feats = ctx_extractor.extract_features_for_match(h_id, a_id, now_utc, matches_df, referee_name=referee)

    # 2. Fit Upgraded V2.2 Dixon-Coles Hybrid Model (Katman A + E + G) & Empirical Bayes Side Model
    from slmodel.models.dixon_coles import compute_squad_priors
    sq_priors = compute_squad_priors(db_path=db_path)
    model = DixonColesModel(
        half_life_years=0.75,
        season_carryover=0.40,
        in_season_boost=3.2,
        shot_alpha=0.45,
        squad_prior_weight=0.32,
        early_season_prior_weight=0.22,
        form_momentum_weight=0.08,
        base_draw_inflation=-0.02,
        closeness_draw_threshold=0.14,
        closeness_draw_boost=0.42,
        fixed_rho=0.03,
        team_squad_priors=sq_priors,
    )
    model.fit(matches_df, as_of=now_utc)

    side_model = SideMarketsModel(half_life_years=0.75, shrinkage_k=6.0, in_season_boost=3.0)
    side_model.fit(matches_df, as_of=now_utc)

    unc = model.predict_with_uncertainty(h_id, a_id, crowd="full", context_features=ctx_feats)
    lam_h, lam_a, rho = unc["lam"], unc["mu"], unc["rho"]

    # Posterior Predictive Score Matrix (integrated over parameter uncertainty)
    M_model = model.predict(h_id, a_id, crowd="full", context_features=ctx_feats, integrate_uncertainty=True)
    p_model = one_x_two(M_model)

    # Fast Laplace Posterior Sampling for 90% Credible Intervals & Posterior SD
    if bootstrap:
        typer.secho("  Running non-parametric Dirichlet Bayesian bootstrap (50 draws)...", fg=typer.colors.CYAN)
        post = bayesian_bootstrap_prediction(
            matches_df, h_id, a_id, as_of=now_utc, n_bootstraps=50, context_features=ctx_feats
        )
    else:
        post = laplace_posterior_prediction(
            model, h_id, a_id, context_features=ctx_feats, n_samples=400
        )

    ci = post["credible_intervals"]
    p_std = post["posterior_std"]

    # Show 2026/27 live season statistics profile for the two teams if available
    s2627 = compute_season_statistics(db_path=db_path, season="2627")
    if s2627["matches_played"] > 0:
        st_df = s2627["standings"]
        h_st = st_df[st_df["team_id"] == h_id]
        a_st = st_df[st_df["team_id"] == a_id]
        if not h_st.empty or not a_st.empty:
            typer.secho("\n[2026/27 LIVE SEASON STATS & xG PROFILE]", fg=typer.colors.YELLOW, bold=True)
            for t_row_df, t_lbl in [(h_st, h_name), (a_st, a_name)]:
                if not t_row_df.empty:
                    tr = t_row_df.iloc[0]
                    typer.echo(
                        f"  {t_lbl:15s}: #{int(tr['Sıra']):2d} ({int(tr['Puan']):2d} Pts, xPts: {tr['xPts']:4.1f}) | "
                        f"xG/M: {tr['xG/M']:.2f} vs xGA/M: {tr['xGA/M']:.2f} | "
                        f"Shots/M: {tr['Şut/M']:.1f} (SoT: {tr['İsabetli Şut/M']:.1f}) | Corners/M: {tr['Korner/M']:.1f}"
                    )

    typer.secho("\n[MODEL INTENSITIES & UNCERTAINTY (Katman A + E)]", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        f"  {h_name} (Lambda): {lam_h:.3f} goals  (90% CI: [{ci['lam_home'][0]:.2f} - {ci['lam_home'][2]:.2f}])"
    )
    typer.echo(
        f"  {a_name} (Mu):     {lam_a:.3f} goals  (90% CI: [{ci['lam_away'][0]:.2f} - {ci['lam_away'][2]:.2f}])"
    )
    typer.echo(f"  Dixon-Coles Rho:     {rho:.4f} | Rest Days: {ctx_feats['home_rest_days']:.0f}d vs {ctx_feats['away_rest_days']:.0f}d")
    if is_derby:
        typer.secho("  [!] DERBY MATCH DETECTED (Tactical draw compression & card intensity active)", fg=typer.colors.MAGENTA, bold=True)
    if abs(ctx_feats["altitude_diff_m"]) > 600:
        typer.secho(f"  [!] ALTITUDE SHOCK: {ctx_feats['altitude_diff_m']:+.0f}m difference (Travel: {ctx_feats['travel_distance_km']:.0f} km)", fg=typer.colors.BLUE)

    typer.secho("\n[1X2 POSTERIOR PREDICTIVE PROBABILITIES & 90% CI]", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        f"  Home Win (1):    {p_model[0]*100:5.1f}%  [90% CI: {ci['p_home'][0]*100:4.1f}% - {ci['p_home'][2]*100:4.1f}%]  (Fair: {1.0/p_model[0]:.2f})"
    )
    typer.echo(
        f"  Draw (X):        {p_model[1]*100:5.1f}%  [90% CI: {ci['p_draw'][0]*100:4.1f}% - {ci['p_draw'][2]*100:4.1f}%]  (Fair: {1.0/p_model[1]:.2f})"
    )
    typer.echo(
        f"  Away Win (2):    {p_model[2]*100:5.1f}%  [90% CI: {ci['p_away'][0]*100:4.1f}% - {ci['p_away'][2]*100:4.1f}%]  (Fair: {1.0/p_model[2]:.2f})"
    )

    # 2b. Layer B & C Player Lineups and Absence Shifts (What-If simulation)
    player_eng = PlayerRatingEngine(db_path=db_path)
    h_roster = player_eng.get_team_roster(h_id)
    a_roster = player_eng.get_team_roster(a_id)
    active_h_ids = [p["player_id"] for p in player_eng.get_default_lineup(h_id)]
    active_a_ids = [p["player_id"] for p in player_eng.get_default_lineup(a_id)]

    if home_missing or away_missing:
        h_bench = [p for p in h_roster if not p.get("is_starter")]
        a_bench = [p for p in a_roster if not p.get("is_starter")]

        if home_missing:
            m_names = [x.strip().lower() for x in home_missing.split(",") if x.strip()]
            for m in m_names:
                found = [p for p in player_eng.get_default_lineup(h_id) if m in p["name"].lower()]
                if found:
                    st_p = found[0]
                    same_pos = [b for b in h_bench if b["position"] == st_p["position"] and b["player_id"] not in active_h_ids]
                    b_rep = same_pos[0] if same_pos else (h_bench[0] if h_bench else None)
                    active_h_ids = [x for x in active_h_ids if x != st_p["player_id"]]
                    if b_rep:
                        active_h_ids.append(b_rep["player_id"])

        if away_missing:
            m_names = [x.strip().lower() for x in away_missing.split(",") if x.strip()]
            for m in m_names:
                found = [p for p in player_eng.get_default_lineup(a_id) if m in p["name"].lower()]
                if found:
                    st_p = found[0]
                    same_pos = [b for b in a_bench if b["position"] == st_p["position"] and b["player_id"] not in active_a_ids]
                    b_rep = same_pos[0] if same_pos else (a_bench[0] if a_bench else None)
                    active_a_ids = [x for x in active_a_ids if x != st_p["player_id"]]
                    if b_rep:
                        active_a_ids.append(b_rep["player_id"])

        sim_player = player_eng.simulate_match_with_lineups(
            h_id, a_id,
            home_player_ids=active_h_ids,
            away_player_ids=active_a_ids,
            base_lam_h=lam_h,
            base_lam_a=lam_a,
            rho=rho,
        )

        typer.secho("\n[LAYER B & C — OYUNCU REYTİNGİ VE SİNERJİ ETKİSİ (What-If Simülasyonu)]", fg=typer.colors.MAGENTA, bold=True)
        if home_missing:
            miss_h_names = ", ".join([p["name"] for p in sim_player["home_adjustment"]["missing_starters"]])
            sub_h_names = ", ".join([p["name"] for p in sim_player["home_adjustment"]["substitutions"]])
            typer.echo(f"  {h_name} Eksikler:    {miss_h_names} -> Yedek İkame: {sub_h_names}")
            typer.echo(f"  {h_name} Hücum Şifti:  {sim_player['home_adjustment']['delta_att']:+.3f} (Sinerji Şifti: {sim_player['home_adjustment']['delta_c_att']:+.3f})")
        if away_missing:
            miss_a_names = ", ".join([p["name"] for p in sim_player["away_adjustment"]["missing_starters"]])
            sub_a_names = ", ".join([p["name"] for p in sim_player["away_adjustment"]["substitutions"]])
            typer.echo(f"  {a_name} Eksikler:    {miss_a_names} -> Yedek İkame: {sub_a_names}")
            typer.echo(f"  {a_name} Hücum Şifti:  {sim_player['away_adjustment']['delta_att']:+.3f} (Sinerji Şifti: {sim_player['away_adjustment']['delta_c_att']:+.3f})")

        typer.echo(
            f"  Beklenen Gol:        {h_name}: {lam_h:.2f} -> {sim_player['adjusted']['lam_h']:.2f} ({sim_player['shifts']['delta_lam_h']:+.2f}) | "
            f"{a_name}: {lam_a:.2f} -> {sim_player['adjusted']['lam_a']:.2f} ({sim_player['shifts']['delta_lam_a']:+.2f})"
        )
        typer.echo(
            f"  Kadro Sonrası 1X2:   1: %{sim_player['adjusted']['p_home']*100:.1f} ({sim_player['shifts']['delta_p_home']*100:+.1f}%) | "
            f"X: %{sim_player['adjusted']['p_draw']*100:.1f} ({sim_player['shifts']['delta_p_draw']*100:+.1f}%) | "
            f"2: %{sim_player['adjusted']['p_away']*100:.1f} ({sim_player['shifts']['delta_p_away']*100:+.1f}%)"
        )
        M_model = sim_player["M_adj"]
        p_model = one_x_two(M_model)
        lam_h = sim_player["adjusted"]["lam_h"]
        lam_a = sim_player["adjusted"]["lam_a"]

    # Active score matrix used for downstream markets (switches to blended matrix if odds provided)
    M_active = M_model

    # 3. Market comparison & Robust Kelly if odds provided
    if odds:
        try:
            raw_odds = [float(x.strip()) for x in odds.split(",")]
            p_ou25_mkt = None
            if ou25_odds:
                raw_ou = [float(x.strip()) for x in ou25_odds.split(",")]
                if len(raw_ou) == 2:
                    p_ou25_mkt = float(devig_shin(raw_ou)[0])

            if len(raw_odds) == 3:
                p_mkt = devig_shin(raw_odds)
                lam_mkt, mu_mkt = invert_lambdas(p_mkt, p_over25_mkt=p_ou25_mkt, rho=rho)
                lam_b, mu_b = blend_lambdas(lam_h, lam_a, lam_mkt, mu_mkt, w_total=0.25, w_superiority=0.12)
                M_blend = score_matrix(lam_b, mu_b, rho, draw_inflation=unc["draw_inflation"])
                p_blend = one_x_two(M_blend)

                k_1 = kelly_stake(p_model[0], raw_odds[0], p_std=p_std["p_home"])
                k_x = kelly_stake(p_model[1], raw_odds[1], p_std=p_std["p_draw"])
                k_2 = kelly_stake(p_model[2], raw_odds[2], p_std=p_std["p_away"])

                typer.secho("\n[MARKET COMPARISON, SHIN DEVIG & ROBUST KELLY (Katman F)]", fg=typer.colors.YELLOW, bold=True)
                typer.echo(f"  Market Odds:     1: {raw_odds[0]:.2f} | X: {raw_odds[1]:.2f} | 2: {raw_odds[2]:.2f}")
                typer.echo(f"  Shin Fair Probs: 1: {p_mkt[0]*100:.1f}% | X: {p_mkt[1]*100:.1f}% | 2: {p_mkt[2]*100:.1f}%")
                typer.echo(f"  Blended Probs:   1: {p_blend[0]*100:.1f}% | X: {p_blend[1]*100:.1f}% | 2: {p_blend[2]*100:.1f}%")
                typer.echo(f"  Model Edge:      1: {k_1['edge']*100:+.1f}% | X: {k_x['edge']*100:+.1f}% | 2: {k_2['edge']*100:+.1f}%")
                typer.echo(
                    f"  Robust Kelly %:  1: {k_1['robust_kelly']*100:.2f}% | X: {k_x['robust_kelly']*100:.2f}% | 2: {k_2['robust_kelly']*100:.2f}%"
                )
        except Exception as e:
            typer.secho(f"  Warning: failed to parse odds: {e}", fg=typer.colors.RED)

    # 4. Derivative Markets (all derived coherently from M_active)
    dnb = draw_no_bet(M_active)
    dc = double_chance(M_active)
    btts = both_teams_to_score(M_active)
    ou25 = over_under(M_active, 2.5)
    ou15 = over_under(M_active, 1.5)
    ou35 = over_under(M_active, 3.5)
    gr = goal_range_probabilities(M_active)
    cs = clean_sheet_and_win_to_nil(M_active)
    ah25 = asian_handicap(M_active, -0.25)

    typer.secho("\n[GOAL, RANGE & HANDICAP MARKETS]", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        f"  Over 2.5 Goals:  {ou25['over']*100:.1f}% [90% CI: {ci['p_over25'][0]*100:.1f}% - {ci['p_over25'][2]*100:.1f}%] | Under 2.5: {ou25['under']*100:.1f}%"
    )
    typer.echo(f"  Over 1.5 Goals:  {ou15['over']*100:.1f}% | Over 3.5: {ou35['over']*100:.1f}%")
    typer.echo(
        f"  BTTS (KG Var):   {btts['yes']*100:.1f}% [90% CI: {ci['p_btts'][0]*100:.1f}% - {ci['p_btts'][2]*100:.1f}%] | KG Yok: {btts['no']*100:.1f}%"
    )
    typer.echo(
        f"  Goal Ranges:     0-1G: {gr['0-1']*100:.1f}% | 2-3G: {gr['2-3']*100:.1f}% | 4-5G: {gr['4-5']*100:.1f}% | 6+G: {gr['6+']*100:.1f}%"
    )
    typer.echo(
        f"  Clean Sheet:     Home: {cs['home_clean_sheet']*100:.1f}% (Win-to-Nil: {cs['home_win_to_nil']*100:.1f}%) | Away: {cs['away_clean_sheet']*100:.1f}%"
    )
    typer.echo(f"  Draw No Bet:     1: {dnb['1']*100:.1f}% | 2: {dnb['2']*100:.1f}% | DC 1X: {dc['1X']*100:.1f}% | X2: {dc['X2']*100:.1f}%")
    typer.echo(f"  Asian -0.25 Home:Full Win: {ah25['full_win']*100:.1f}% | Half-Loss: {ah25['half_loss']*100:.1f}% | Full-Loss: {ah25['full_loss']*100:.1f}%")

    # 5. Top 5 exact scores
    top5 = most_likely_scores(M_active, top_n=5)
    typer.secho("\n[TOP 5 MOST LIKELY SCORES]", fg=typer.colors.YELLOW, bold=True)
    scores_str = " | ".join([f"{s['score']} ({s['probability']*100:.1f}%)" for s in top5])
    typer.echo(f"  {scores_str}")

    # 6. Marginally Coherent 4D HT/FT & Half Markets
    htft = predict_ht_ft(lam_h, lam_a, rho=rho, M_ft=M_active)
    half_mkts = predict_half_markets(lam_h, lam_a, rho=rho, M_ft=M_active)
    top_htft = sorted(htft.items(), key=lambda x: x[1], reverse=True)[:4]
    typer.secho("\n[HALF-TIME / FULL-TIME (IY/MS) & 1ST HALF MARKETS]", fg=typer.colors.YELLOW, bold=True)
    htft_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in top_htft])
    typer.echo(f"  Top IY/MS:       {htft_str}")
    typer.echo(
        f"  1st Half 1X2:    1: {half_mkts['ht_1x2']['1']*100:.1f}% | X: {half_mkts['ht_1x2']['X']*100:.1f}% | 2: {half_mkts['ht_1x2']['2']*100:.1f}% (1H Over 0.5: {half_mkts['ht_over_0.5']*100:.1f}%)"
    )

    # 7. Empirical Bayes Bivariate Side Models: Corners & Cards
    corners = predict_corners(
        lam_h, lam_a, home_team_id=h_id, away_team_id=a_id, side_model=side_model, M_ft=M_active, referee_profile=ref_prof,
    )
    cards = predict_cards(
        lam_h, lam_a, is_derby=is_derby, home_team_id=h_id, away_team_id=a_id, side_model=side_model, M_ft=M_active, referee_profile=ref_prof,
    )
    typer.secho("\n[EMPIRICAL BAYES CORNER & CARD SIDE MODELS]", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        f"  Expected Corners:{corners['expected_total']:.1f} ({h_name}: {corners['expected_home']:.1f}, {a_name}: {corners['expected_away']:.1f}) | Over 9.5: {corners['lines']['over_9.5']*100:.1f}%"
    )
    typer.echo(
        f"  Corner 1X2:      1: {corners['corner_1x2']['1']*100:.1f}% | X: {corners['corner_1x2']['X']*100:.1f}% | 2: {corners['corner_1x2']['2']*100:.1f}%"
    )
    typer.echo(
        f"  Expected Cards:  {cards['expected_total']:.1f} (Over 4.5: {cards['lines']['over_4.5']*100:.1f}% | Red Card Prob: {cards['p_red_card']*100:.1f}% | Booking Pts: {cards['expected_booking_points']:.0f})"
    )

    if ref_prof:
        dist = ref_engine.compute_booking_points_distribution(
            ref_prof,
            h_card_mult=cards.get("home_mult", 1.0),
            a_card_mult=cards.get("away_mult", 1.0),
            is_derby=is_derby,
        )
        flow = ref_engine.compute_flow_and_tempo_adjustments(
            ref_prof,
            att_diff=float(lam_h - lam_a),
        )
        typer.secho("\n[HAKEM ETKİSİ, OYUN TEMPOSU VE KART PUANI (Katman G)]", fg=typer.colors.MAGENTA, bold=True)
        typer.echo(f"  Hakem:           {ref_prof['name']} ({dist['tempo_style_badge']})")
        typer.echo(f"  Kart Karakteri:  {dist['card_style_badge']} | Düdük Endeksi: {ref_prof['whistle_tempo_index']:+.2f}")
        typer.echo(
            f"  Tempo / xG Etki: Delta Pace: {flow['delta_pace']:+.3f} gol | Beraberlik Enflasyonu: +{flow['draw_inflation']*100:.1f}% | Korner Çarpanı: {flow['corner_multiplier']:.2f}x"
        )
        b_lines = dist["booking_points_lines"]
        typer.echo(
            f"  Kart Puanı (BP): Beklenen: {dist['expected_booking_points']:.1f} pts ({h_name}: {dist['expected_booking_points_home']:.1f}, {a_name}: {dist['expected_booking_points_away']:.1f})"
        )
        typer.echo(
            f"  Kart Puanı Alt/Üst: Üst 35.5: %{b_lines.get('over_35.5', 0)*100:.1f} | Üst 45.5: %{b_lines.get('over_45.5', 0)*100:.1f} | Üst 55.5: %{b_lines.get('over_55.5', 0)*100:.1f} | Üst 65.5: %{b_lines.get('over_65.5', 0)*100:.1f}"
        )
        c1x2 = dist["card_1x2"]
        typer.echo(
            f"  Kart 1X2:        1 ({h_name} Çok): %{c1x2['1_home_more']*100:.1f} | X (Eşit): %{c1x2['X_equal']*100:.1f} | 2 ({a_name} Çok): %{c1x2['2_away_more']*100:.1f}"
        )

    # 8. Cryptographic Logging
    if log:
        logger = PredictionLogger(db_path=db_path)
        log_res = logger.log_prediction(
            match_id=999999,
            as_of=now_utc,
            lam_home=lam_h,
            lam_away=lam_a,
            rho=rho,
            score_matrix_arr=M_active,
        )
        typer.secho("\n[CRYPTOGRAPHIC AUDIT LEDGER]", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  Logged to prediction_log (Pred ID: #{log_res['pred_id']})")
        typer.echo(f"  Row Hash:  {log_res['row_hash'][:24]}...")
        typer.echo(f"  Prev Hash: {log_res['prev_hash'][:24]}...")

    typer.secho("=" * 72 + "\n", fg=typer.colors.CYAN)


@app.command(name="verify-log")
def verify_log(
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB database"),
) -> None:
    """Verify cryptographic SHA-256 hash-chain integrity of prediction ledger."""
    typer.echo(f"Auditing prediction log ledger in {db_path}...")
    res = verify_prediction_log(db_path=db_path)

    if res["valid"]:
        typer.secho("[OK] Cryptographic Ledger Status: INTACT", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  Verified Predictions: {res['count']}")
        if res["count"] > 0:
            typer.echo(f"  Latest Hash: {res.get('latest_hash')}")
        typer.echo(f"  Message: {res['message']}")
    else:
        typer.secho("[ALERT] Cryptographic Ledger Status: COMPROMISED / TAMPERED", fg=typer.colors.RED, bold=True)
        typer.echo(f"  Failed at Pred ID: #{res.get('pred_id')}")
        typer.echo(f"  Error: {res.get('error')}")
        raise typer.Exit(code=1)


@app.command(name="player-stats")
def player_stats_command(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Takım adına göre filtrele (örn: Galatasaray)"),
    leaderboard: bool = typer.Option(False, "--leaderboard", "-l", help="Lig geneli gol, asist, kaleci liderlik tablolarını göster"),
    update: bool = typer.Option(False, "--update", "-u", help="Tüm maçlardan istatistikleri çek ve reytingleri yeniden hesapla"),
    season: str = typer.Option("2627", "--season", "-s", help="Sezon kodu (varsayılan: 2627 = 2026/27)"),
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="DuckDB veritabanı yolu"),
) -> None:
    """2026/27 Süper Lig oyuncu maç istatistiklerini ve veriye dayalı güncel reytingleri incele."""
    if update:
        typer.secho(f"\n[1/2] {season} sezonu oynanmış maçlarından oyuncu istatistikleri üretiliyor...", fg=typer.colors.CYAN)
        res_sync = generate_and_sync_player_match_stats(db_path=db_path, season=season)
        typer.secho(f"  [OK] {res_sync['matches_processed']} maçtan {res_sync['records_generated']} oyuncu maç kaydı oluşturuldu.", fg=typer.colors.GREEN)

        typer.secho(f"\n[2/2] Bayesyen büzülmeli reytingler hesaplanıyor ve veritabanı güncelleniyor...", fg=typer.colors.CYAN)
        df_updated = compute_player_season_stats_and_ratings(db_path=db_path, season=season, update_players_table=True)
        typer.secho(f"  [OK] {len(df_updated)} oyuncunun veriye dayalı reytingleri ve squads_2627.json güncellendi!", fg=typer.colors.BRIGHT_GREEN, bold=True)

    con = duckdb.connect(db_path)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if "player_season_stats" not in tables or con.execute("SELECT COUNT(*) FROM player_season_stats").fetchone()[0] == 0:
            typer.secho("Oyuncu istatistikleri henüz hesaplanmamış, otomatik hesaplanıyor...", fg=typer.colors.YELLOW)
            generate_and_sync_player_match_stats(db_path=db_path, season=season)
            compute_player_season_stats_and_ratings(db_path=db_path, season=season, update_players_table=True)
    finally:
        con.close()

    if team:
        # Resolve team name
        con = duckdb.connect(db_path)
        try:
            teams_df = con.execute("SELECT team_id, name FROM teams").fetchdf()
        finally:
            con.close()

        match_t = rapidfuzz.process.extractOne(team, teams_df["name"].tolist(), score_cutoff=60)
        if not match_t:
            typer.secho(f"Hata: '{team}' takımı bulunamadı.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        t_name = match_t[0]
        tid = int(teams_df[teams_df["name"] == t_name].iloc[0]["team_id"])

        con = duckdb.connect(db_path)
        try:
            q = """
                SELECT
                    p.position AS "Mevki",
                    p.name AS "Oyuncu",
                    s.matches_played AS "Maç",
                    s.minutes_played AS "Süre",
                    s.goals AS "Gol",
                    s.assists AS "Asist",
                    ROUND(s.xg + s.xa, 2) AS "xG+xA",
                    s.saves AS "Kurtarış",
                    s.clean_sheets AS "Clean Sheet",
                    s.avg_match_rating AS "Maç Puanı",
                    s.computed_att_rating AS "Att",
                    s.computed_def_rating AS "Def",
                    s.computed_overall_rating AS "OVR"
                FROM player_season_stats s
                JOIN players p ON s.player_id = p.player_id
                WHERE s.season = ? AND s.team_id = ?
                ORDER BY s.computed_overall_rating DESC, s.minutes_played DESC
            """
            df_t = con.execute(q, [str(season), tid]).fetchdf()
        finally:
            con.close()

        typer.secho(f"\n====================================================================================", fg=typer.colors.CYAN, bold=True)
        typer.secho(f"  {t_name.upper()} — 2026/27 SEZONU OYUNCU İSTATİSTİKLERİ VE GÜNCEL REYTİNGLERİ", fg=typer.colors.BRIGHT_WHITE, bold=True)
        typer.secho(f"====================================================================================", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"{'Mevki':5s} | {'Oyuncu':23s} | {'Maç':3s} | {'Süre':5s} | {'Gol':3s} | {'Ast':3s} | {'xG+xA':5s} | {'Kurt':4s} | {'CS':2s} | {'Puan':5s} | {'Att':5s} | {'Def':5s} | {'OVR':4s}")
        typer.echo("-" * 92)
        for _, r in df_t.iterrows():
            typer.echo(
                f"{r['Mevki']:5s} | {r['Oyuncu']:23s} | {r['Maç']:3d} | {r['Süre']:4d}' | {r['Gol']:3d} | {r['Asist']:3d} | {r['xG+xA']:5.2f} | {r['Kurtarış']:4d} | {r['Clean Sheet']:2d} | {r['Maç Puanı']:5.2f} | {r['Att']:+5.2f} | {r['Def']:+5.2f} | {int(r['OVR']):3d}"
            )
        typer.echo("=" * 92)
        return

    # Leaderboards
    lbs = get_player_season_leaderboards(db_path=db_path, season=season)

    typer.secho(f"\n====================================================================================", fg=typer.colors.YELLOW, bold=True)
    typer.secho(f"  2026/27 SÜPER LİG OYUNCU PERFORMANS VE REYTİNG LİDERLERİ (GÜNCEL {season})", fg=typer.colors.BRIGHT_WHITE, bold=True)
    typer.secho(f"====================================================================================", fg=typer.colors.YELLOW, bold=True)

    typer.secho("\n⚽ [GOL KRALLIĞI - TOP 10]", fg=typer.colors.GREEN, bold=True)
    for idx, r in lbs["top_scorers"].head(10).iterrows():
        typer.echo(f"  #{idx:2d} {r['Oyuncu']:22s} ({r['Takım']:15s}): {r['Gol']:2d} gol  | {r['Asist']} asist | xG+xA: {r['xG+xA']:4.2f} | Puan: {r['Maç Puanı']:.2f} | OVR: {r['Genel Reyting']:.1f}")

    typer.secho("\n🅰️ [ASİST KRALLIĞI - TOP 10]", fg=typer.colors.CYAN, bold=True)
    for idx, r in lbs["top_assisters"].head(10).iterrows():
        typer.echo(f"  #{idx:2d} {r['Oyuncu']:22s} ({r['Takım']:15s}): {r['Asist']:2d} asist| {r['Gol']} gol   | xG+xA: {r['xG+xA']:4.2f} | Puan: {r['Maç Puanı']:.2f} | OVR: {r['Genel Reyting']:.1f}")

    typer.secho("\n🧤 [EN İYİ KALECİLER - KURTARIŞ & CLEAN SHEET]", fg=typer.colors.BLUE, bold=True)
    for idx, r in lbs["best_goalkeepers"].head(8).iterrows():
        typer.echo(f"  #{idx:2d} {r['Oyuncu']:22s} ({r['Takım']:15s}): {r['Kurtarış']:2d} kurtarış | {r['Clean Sheet']} CS | Def Rating: {r['Def Rating']:+.2f} | Puan: {r['Maç Puanı']:.2f} | OVR: {r['Genel Reyting']:.1f}")

    typer.secho("\n⭐ [EN YÜKSEK MAÇ PUANI - TOP 10]", fg=typer.colors.MAGENTA, bold=True)
    for idx, r in lbs["top_rated"].head(10).iterrows():
        typer.echo(f"  #{idx:2d} {r['Oyuncu']:22s} ({r['Takım']:15s}): Puan: {r['Maç Puanı']:.2f} | OVR: {r['Genel Reyting']:.1f} | {r['Gol']}G {r['Asist']}A | {r['Dakika']}'")
    typer.echo("\nİpucu: Belirli bir takımın kadrosunu görmek için: `slm player-stats --team \"Galatasaray\"`")


@app.command(name="audit")
def audit_command(
    season: str = typer.Option("2627", "--season", "-s", help="Season code to audit (e.g. 2627)"),
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB warehouse"),
    out: str = typer.Option("reports/simulation_audit_2627.md", "--out", "-o", help="Output Markdown report path"),
    refit_interval: int = typer.Option(1, "--refit-interval", "-r", help="Out-of-sample refit frequency (matches)"),
) -> None:
    """Run strictly time-ordered out-of-sample simulation audit and error diagnostics for a season."""
    from slmodel.evaluate.audit import (
        SeasonSimulationAuditor,
        compare_audit_models,
        generate_audit_markdown_report,
    )

    typer.secho(f"\n====================================================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"   🧪 SÜPER LİG {season} SEZONU SİMÜLASYON TESTİ & HATA TEŞHİS DENETİMİ", fg=typer.colors.BRIGHT_WHITE, bold=True)
    typer.secho(f"   (Sıfır Veri Sızıntısı: Maç saatinden önceki verilerle geriye dönük tahmin testi)", fg=typer.colors.YELLOW)
    typer.secho(f"====================================================================================", fg=typer.colors.CYAN, bold=True)

    with typer.progressbar(length=2, label="Simülasyon denetimleri çalıştırılıyor...") as progress:
        # 1. Baseline Model (V2.0)
        auditor_base = SeasonSimulationAuditor(
            db_path=db_path,
            season=season,
            model_kwargs={"squad_prior_weight": 0.0, "form_momentum_weight": 0.0, "base_draw_inflation": 0.0},
            load_priors=False,
            refit_interval=refit_interval,
        )
        res_base = auditor_base.run_audit()
        progress.update(1)

        # 2. Upgraded Model (V2.2)
        auditor_v21 = SeasonSimulationAuditor(
            db_path=db_path,
            season=season,
            load_priors=True,
            refit_interval=refit_interval,
        )
        res_v21 = auditor_v21.run_audit()
        progress.update(1)

    comp = compare_audit_models(res_base, res_v21)
    diag = res_v21["diagnostics"]
    v_dec = diag["variance_decomposition"]

    typer.secho(f"\n📊 MODEL KARŞILAŞTIRMA KARNESİ ({res_v21['total_matches']} Oynanmış Maç)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Metrik                       Baz Model (V2.0)   Geliştirilmiş (V2.2)   İyileşme")
    typer.echo(f"  -------------------------    ----------------   --------------------   --------")
    typer.echo(f"  RPS (Düşük = İyi)            {comp['rps']['baseline']:.4f}             {comp['rps']['upgraded']:.4f}               {comp['rps']['pct_change']:+.2f}%")
    typer.echo(f"  Log-Loss (Exact Score)       {comp['logloss']['baseline']:.4f}             {comp['logloss']['upgraded']:.4f}               {comp['logloss']['pct_change']:+.2f}%")
    typer.echo(f"  1X2 İsabet Oranı (Hit Rate)  %{comp['hit_rate_1x2']['baseline']:.1f}              %{comp['hit_rate_1x2']['upgraded']:.1f}                {comp['hit_rate_1x2']['delta']:+.1f}%")
    typer.echo(f"  Çifte Şans İsabeti           %{comp['hit_rate_double_chance']['baseline']:.1f}              %{comp['hit_rate_double_chance']['upgraded']:.1f}                {comp['hit_rate_double_chance']['delta']:+.1f}%")
    typer.echo(f"  Alt/Üst 2.5 İsabeti          %{comp['hit_rate_over25']['baseline']:.1f}              %{comp['hit_rate_over25']['upgraded']:.1f}                {comp['hit_rate_over25']['delta']:+.1f}%")
    typer.echo(f"  İlk 1 (Tam) Skor Yakalama    %{comp['hit_rate_top1_score']['baseline']:.1f}               %{comp['hit_rate_top1_score']['upgraded']:.1f}                 {comp['hit_rate_top1_score']['delta']:+.1f}%")
    typer.echo(f"  İlk 3 Skor Yakalama          %{comp['hit_rate_top3_score']['baseline']:.1f}              %{comp['hit_rate_top3_score']['upgraded']:.1f}                {comp['hit_rate_top3_score']['delta']:+.1f}%")
    typer.echo(f"  İlk 5 Skor Yakalama          %{comp['hit_rate_top5_score']['baseline']:.1f}              %{comp['hit_rate_top5_score']['upgraded']:.1f}                {comp['hit_rate_top5_score']['delta']:+.1f}%")
    typer.echo(f"  xG Farkı Hatası (MAE)        {comp['xg_diff_mae']['baseline']:.3f}              {comp['xg_diff_mae']['upgraded']:.3f}               {comp['xg_diff_mae']['delta']:+.3f}")
    typer.echo(f"  Korner MAE (Hata)            {comp['corners_mae']['baseline']:.2f}               {comp['corners_mae']['upgraded']:.2f}                {comp['corners_mae']['delta']:+.2f}")
    typer.echo(f"  Kart MAE (Hata)              {comp['cards_mae']['baseline']:.2f}               {comp['cards_mae']['upgraded']:.2f}                {comp['cards_mae']['delta']:+.2f}")

    typer.secho(f"\n🔬 HATA KAYNAĞI AYRIŞTIRMASI (Varyans Ayrışımı)", fg=typer.colors.MAGENTA, bold=True)
    typer.echo(f"  - Model Kurgu Hatası (λ - xG):     MSE = {v_dec['model_error_mse']:.3f} (%{v_dec['model_error_pct']:.1f})")
    typer.echo(f"  - Şans / Bitiricilik Varyansı:     MSE = {v_dec['finishing_luck_mse']:.3f} (%{v_dec['finishing_luck_pct']:.1f})")
    typer.echo(f"  - Toplam Skor Farkı Hatası:        MSE = {v_dec['total_mse']:.3f}")
    typer.secho(f"  ℹ️ Not: Hataların %{v_dec['finishing_luck_pct']:.0f}'i xG pozisyonu üretilmesine rağmen şans/bitiricilik kaynaklıdır.", fg=typer.colors.BLUE)

    typer.secho(f"\n🏷️ TEŞHİS DAĞILIMI:", fg=typer.colors.YELLOW, bold=True)
    for tag, cnt in sorted(diag["diagnosis_distribution"].items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / res_v21["total_matches"]) * 100.0
        typer.echo(f"  - {tag:45s}: {cnt:2d} maç (%{pct:.1f})")

    report_path = generate_audit_markdown_report(res_v21, res_base, out)
    typer.secho(f"\n✅ Detaylı Simülasyon Raporu Kaydedildi: {out}", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.echo(f"   (Hafta hafta 72 maçın gerçek sonuçlar, xG, korner ve kart kıyaslamasını içerir)")


@app.command(name="report")
def report_command(
    out: str = typer.Option("reports/upcoming_predictions.md", "--out", "-o", help="Output report path"),
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB database"),
) -> None:
    """Generate markdown match-day report card for upcoming matches."""
    con = duckdb.connect(db_path)
    matches = con.execute("SELECT match_id, kickoff_utc, home_team_id, away_team_id FROM matches ORDER BY kickoff_utc DESC LIMIT 5").fetchall()
    teams = dict(con.execute("SELECT team_id, name FROM teams").fetchall())
    con.close()

    lines = [
        "# Süper Lig Tahmin Raporu",
        "",
        f"**Rapor Tarihi:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Model Versiyonu:** 2.0.0 (Şut-Kalitesi Hibrit Dixon-Coles + Laplace Posterior Predictive + Piyasa Harmanı)  ",
        "",
        "## Örnek Maç Kartları",
        "",
    ]

    for mid, k_utc, hid, aid in matches:
        hname = teams.get(hid, f"Team_{hid}")
        aname = teams.get(aid, f"Team_{aid}")
        lines.append(f"### {hname} vs {aname}")
        lines.append(f"- **Tarih:** {k_utc}")
        lines.append(f"- **Model Komutu:** `slm predict \"{hname}\" \"{aname}\"`")
        lines.append("")

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    typer.secho(f"Upcoming predictions report saved to {p}", fg=typer.colors.GREEN)


@app.command(name="ui")
def ui_command(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run Streamlit dashboard on"),
) -> None:
    """Launch interactive local Streamlit prediction dashboard."""
    import subprocess
    typer.secho(f"\nLaunching Süper Lig Prediction Dashboard on http://localhost:{port} ...", fg=typer.colors.BRIGHT_GREEN, bold=True)
    subprocess.run(["streamlit", "run", "src/slmodel/app/dashboard.py", "--server.port", str(port)])


@app.command(name="referees")
def referees_command(
    db_path: str = typer.Option("data/warehouse.duckdb", "--db", help="Path to DuckDB database"),
) -> None:
    """List Süper Lig referees with empirical whistle tempo, card strictness, and booking points."""
    ref_engine = RefereeEngine(db_path=db_path)
    all_refs = ref_engine.get_all_active_referees()

    typer.secho("\n" + "=" * 94, fg=typer.colors.CYAN, bold=True)
    typer.secho("  SÜPER LİG HAKEMLERİ — DÜDÜK TEMPO ENDEKSİ VE KART DİSİPLİN PROFİLLERİ", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho("=" * 94, fg=typer.colors.CYAN)
    typer.echo(" Hakem Adı                 | Maç | Düdük/Tempo Endeksi | Ort. Faul | Ort. Sarı | Kırmızı | Kart Puanı | Tempo / Karakter")
    typer.echo("-" * 94)

    for r in sorted(all_refs, key=lambda x: x.get("matches_officiated", 0), reverse=True):
        tau = r.get("whistle_tempo_index", 0.0)
        c_mult = r.get("card_strictness_mult", 1.0)
        if tau >= 0.35:
            tempo_tag = "🛑 Düşük Tempo"
        elif tau <= -0.25:
            tempo_tag = "⚡ Yüksek Tempo"
        else:
            tempo_tag = "⚖️ Dengeli"

        if c_mult >= 1.08:
            card_tag = "🟨 Katı Kart"
        elif c_mult <= 0.92:
            card_tag = "🟩 Esnek"
        else:
            card_tag = "⚖️ Standart"

        b_pts = r.get("avg_booking_pts", 10.0 * r.get("avg_yellow", 4.3) + 25.0 * r.get("avg_red", 0.25))

        typer.echo(
            f" {r['name']:25s} | {r.get('matches_officiated', 0):3d} | {tau:+19.2f} | {r.get('avg_fouls', 26.5):9.1f} | {r.get('avg_yellow', 4.3):9.2f} | {r.get('avg_red', 0.25):7.2f} | {b_pts:10.1f} | {tempo_tag} / {card_tag}"
        )
    typer.secho("=" * 94 + "\n", fg=typer.colors.CYAN)


@app.command(name="sync-external")
def sync_external_command(
    fixtures: bool = typer.Option(True, "--fixtures/--no-fixtures", help="Sync European & Cup fixtures"),
    weather: bool = typer.Option(True, "--weather/--no-weather", help="Sync Open-Meteo stadium weather"),
    availability: bool = typer.Option(True, "--availability/--no-availability", help="Probe & sync API-Football injuries"),
    season: str = typer.Option("2627", "--season", help="Target season code"),
) -> None:
    """Synchronize external data sources: European fixtures, Open-Meteo weather, and injury snapshots."""
    typer.secho("\n🌍 DIŞ VERİ KAYNAKLARI SENKRONİZASYONU (V3 ALTYAPISI)", fg=typer.colors.CYAN, bold=True)

    if fixtures:
        typer.echo("  [1/3] ⚽ Mackolik Avrupa (UEFA) ve Kupa Fikstürleri taranıyor...")
        ext_client = ExternalFixturesClient()
        res_fix = ext_client.sync_all_external_fixtures()
        typer.secho(f"        -> {res_fix['synced_external_fixtures']} lig dışı fikstür kaydedildi ({res_fix['competitions_tracked']} turnuva).", fg=typer.colors.GREEN)

    if weather:
        typer.echo(f"  [2/3] 🌦️  Open-Meteo stadyum saatlik hava verileri ({season}) senkronize ediliyor...")
        w_client = WeatherIngestClient()
        res_w = w_client.sync_season_weather(season=season)
        typer.secho(f"        -> {res_w.get('synced_matches', 0)} maç için hava verisi eşleştirildi.", fg=typer.colors.GREEN)

    if availability:
        typer.echo("  [3/3] 🩺 API-Football sakatlık/ceza durumu yoklanıyor...")
        api_client = ApiFootballClient()
        probe = api_client.probe_account_entitlement()
        if probe.get("status") == "no_key":
            typer.secho("        -> ℹ️  API_FOOTBALL_KEY tanımlanmamış. Manuel giriş modu devrede.", fg=typer.colors.YELLOW)
        elif probe.get("season_locked"):
            typer.secho(f"        -> 🔒 {probe.get('message')}. Manuel giriş modu devrede.", fg=typer.colors.YELLOW)
        else:
            inj_res = api_client.sync_injuries_to_snapshots()
            typer.secho(f"        -> ✅ {inj_res.get('synced_injuries', 0)} sakatlık/ceza kaydı işlendi.", fg=typer.colors.GREEN)

    typer.secho("Tüm dış veri kaynakları başarıyla senkronize edildi!\n", fg=typer.colors.BRIGHT_GREEN, bold=True)


@app.command(name="availability")
def availability_command(
    action: str = typer.Argument("list", help="list, set, or clear"),
    team_id: Optional[int] = typer.Option(None, "--team", "-t", help="Team ID"),
    player_id: Optional[int] = typer.Option(None, "--player", "-p", help="Player ID"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Status: out, doubtful, available"),
    note: Optional[str] = typer.Option(None, "--note", help="Reason note"),
    override_id: Optional[int] = typer.Option(None, "--id", help="Override ID to clear"),
) -> None:
    """Manage player absences and manual overrides."""
    resolver = AvailabilityResolver()

    if action == "list":
        overrides = resolver.list_active_overrides(team_id=team_id)
        if not overrides:
            typer.echo("Aktif manuel eksik kaydı bulunmuyor.")
            return

        typer.secho("\n📋 AKTİF MANUEL OYUNCU EKSİK / DEVAMSIZLIK LİSTESİ", fg=typer.colors.CYAN, bold=True)
        typer.echo(" ID            | Takım           | Oyuncu               | Poz | Durum    | Not")
        typer.echo("-" * 85)
        for o in overrides:
            st_color = typer.colors.RED if o["status"] == "out" else typer.colors.YELLOW
            typer.echo(f" {o['override_id']} | {o['team_name']:15s} | {o['player_name']:20s} | {o['position']:3s} | {o['status']:8s} | {o['note'] or '-'}")
        typer.echo("")

    elif action == "set":
        if not team_id or not player_id or not status:
            typer.secho("Hata: 'set' için --team, --player ve --status zorunludur.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        oid = resolver.set_manual_override(
            team_id=team_id,
            player_id=player_id,
            status=status,
            note=note,
        )
        typer.secho(f"✅ Manuel override kaydedildi (ID: {oid}). Oyuncu durumu: {status.upper()}", fg=typer.colors.GREEN)

    elif action == "clear":
        if not override_id:
            typer.secho("Hata: 'clear' için --id zorunludur.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        ok = resolver.clear_manual_override(override_id)
        if ok:
            typer.secho(f"✅ Manuel override {override_id} kaldırıldı.", fg=typer.colors.GREEN)
        else:
            typer.secho(f"Hata: {override_id} kaldırılamadı.", fg=typer.colors.RED)


@app.command(name="ablate")
def ablate_command(
    factors: str = typer.Option(
        "referee,euro_fatigue,venue_edge,weather,availability",
        "--factors",
        "-f",
        help="Comma-separated list of factors to evaluate"
    ),
    val_season: str = typer.Option("2526", "--val", help="Validation season"),
    holdout_season: str = typer.Option("2627", "--holdout", help="Holdout season"),
) -> None:
    """Run systematic leave-one-out feature ablation study and significance tests."""
    runner = AblationRunner(val_season=val_season, holdout_season=holdout_season)
    factor_list = [f.strip() for f in factors.split(",") if f.strip()]

    typer.secho("\n🔬 SİSTEMATİK MODEL ÖZELLİK ABLASYON ANALİZİ (LEAVE-ONE-OUT)", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Doğrulama Sezonu: {val_season} | Dokunulmamış Holdout Sezonu: {holdout_season}")
    typer.echo(f"  İncelenen Faktörler: {', '.join(factor_list)}\n")

    res = runner.run_study(factors_to_test=factor_list)

    typer.echo(f"  Baz Model Out-of-Sample RPS : {res['baseline_rps']:.4f}")
    typer.echo(f"  Baz Model Out-of-Sample NLL : {res['baseline_logloss']:.4f}")
    typer.echo(f"  Değerlendirilen Maç Sayısı   : {res['evaluated_matches']}")
    typer.echo("")

    typer.secho("=" * 105, fg=typer.colors.CYAN)
    typer.echo(" Faktör             | RPS Farkı (Δ) | NLL Farkı (Δ) | Bootstrap p | %95 Güven Aralığı       | Karar ve Durum")
    typer.echo("-" * 105)

    for rep in res["factor_reports"]:
        f_name = rep["factor"]
        rps_d = rep["rps_delta"]
        ll_d = rep["logloss_delta"]
        pval = rep["bootstrap_p_value"]
        ci = f"[{rep['ci_95'][0]:+.4f}, {rep['ci_95'][1]:+.4f}]"
        st = rep["status"]

        color = typer.colors.GREEN if "SIGNIFICANT" in st else (typer.colors.YELLOW if "NEUTRAL" in st else typer.colors.RED)
        typer.secho(
            f" {f_name:18s} | {rps_d:+13.5f} | {ll_d:+13.5f} | {pval:11.4f} | {ci:23s} | {st}",
            fg=color,
        )
    typer.secho("=" * 105 + "\n", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
