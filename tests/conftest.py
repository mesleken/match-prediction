"""Pytest configuration and session fixtures for clean CI runs and data warehouse isolation."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import duckdb
import pytest

from slmodel.data.seed_players import seed_players_database


@pytest.fixture(scope="session", autouse=True)
def ensure_warehouse_exists():
    """
    Ensure that data/warehouse.duckdb exists with valid schema, teams, players, and match records.
    Allows clean clones (where warehouse.duckdb is gitignored) and CI runners to pass 100% of tests.
    """
    db_path = Path("data/warehouse.duckdb")
    if db_path.exists():
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # 1. Apply schema DDL
    schema_file = Path("src/slmodel/etl/schema.sql")
    if schema_file.exists():
        con.execute(schema_file.read_text(encoding="utf-8"))

    # 2. Seed 18 Süper Lig Teams
    teams_data = [
        (1, "Galatasaray"), (2, "Fenerbahçe"), (3, "Beşiktaş"), (4, "Trabzonspor"),
        (5, "Başakşehir"), (6, "Kasımpaşa"), (7, "Sivasspor"), (8, "Alanyaspor"),
        (9, "Rizespor"), (10, "Antalyaspor"), (11, "Gaziantep FK"), (12, "Konyaspor"),
        (13, "Kayserispor"), (14, "Samsunspor"), (15, "Bodrum FK"), (16, "Eyüpspor"),
        (17, "Göztepe"), (18, "Hatayspor")
    ]
    con.executemany("INSERT OR IGNORE INTO teams (team_id, name) VALUES (?, ?)", teams_data)

    # 3. Seed Stadiums
    for tid, tname in teams_data:
        con.execute(
            "INSERT OR IGNORE INTO stadiums (stadium_id, name, city, lat, lon, altitude_m) VALUES (?, ?, ?, ?, ?, ?)",
            [tid, f"{tname} Stadyumu", "Istanbul", 41.0, 29.0, 50.0]
        )

    # 4. Seed Referees
    con.execute("""
        INSERT OR IGNORE INTO referees (referee_id, name, matches_officiated, avg_fouls, avg_yellow, avg_red, avg_penalties, home_foul_ratio, whistle_tempo_index, card_strictness_mult, red_card_mult, home_card_bias)
        VALUES 
            (1, 'Halil Umut Meler', 150, 26.5, 4.8, 0.22, 0.35, 0.48, 0.05, 1.10, 1.05, 0.02),
            (2, 'Atilla Karaoğlan', 120, 28.0, 5.2, 0.28, 0.40, 0.49, 0.08, 1.15, 1.10, 0.03),
            (3, 'Ali Şansalan', 110, 25.0, 4.5, 0.18, 0.30, 0.47, -0.02, 0.95, 0.90, -0.01)
    """)

    # 5. Populate synthetic matches across seasons (2122 to 2627)
    base_date = datetime(2021, 8, 15, 18, 0, tzinfo=timezone.utc)
    match_id = 1
    snapshot_id = 1
    seasons = ["2122", "2223", "2324", "2425", "2526", "2627"]

    matches_rows = []
    stats_rows = []
    odds_rows = []
    weather_rows = []

    for s_idx, season in enumerate(seasons):
        for md in range(1, 10):
            for t_idx in range(0, 18, 2):
                h_id = teams_data[t_idx][0]
                a_id = teams_data[t_idx + 1][0]
                m_date = base_date + timedelta(days=s_idx * 365 + md * 7 + t_idx)
                
                hg = int((h_id + md) % 4)
                ag = int((a_id + md) % 3)
                ref_name = "Halil Umut Meler" if (match_id % 3 == 0) else ("Atilla Karaoğlan" if match_id % 3 == 1 else "Ali Şansalan")

                matches_rows.append((
                    match_id, "super_lig", season, md, m_date,
                    h_id, a_id, hg, ag, 0, 0, ref_name, h_id, "played", "full", False, f"hash_{match_id}"
                ))

                # Home stats
                stats_rows.append((match_id, h_id, True, 12, 5, 6, 14, 2, 0, 1.45))
                # Away stats
                stats_rows.append((match_id, a_id, False, 9, 3, 4, 12, 3, 0, 0.95))

                # Closing odds
                odds_rows.append((snapshot_id, match_id, "bet365", "1x2", "1", None, 1.85, m_date, "closing", "test"))
                odds_rows.append((snapshot_id + 1, match_id, "bet365", "1x2", "X", None, 3.50, m_date, "closing", "test"))
                odds_rows.append((snapshot_id + 2, match_id, "bet365", "1x2", "2", None, 4.20, m_date, "closing", "test"))
                odds_rows.append((snapshot_id + 3, match_id, "bet365", "ou", "over", 2.5, 1.80, m_date, "closing", "test"))
                odds_rows.append((snapshot_id + 4, match_id, "bet365", "ou", "under", 2.5, 2.05, m_date, "closing", "test"))
                snapshot_id += 5

                # Weather
                weather_rows.append((match_id, 18.0, 0.0, 12.0))
                match_id += 1

    con.executemany("""
        INSERT INTO matches (match_id, league, season, matchday, kickoff_utc, home_team_id, away_team_id, home_goals, away_goals, ht_home_goals, ht_away_goals, referee, stadium_id, status, crowd, is_derby, src_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, matches_rows)

    con.executemany("""
        INSERT INTO team_match_stats (match_id, team_id, is_home, shots, shots_on_target, corners, fouls, yellow, red, xg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, stats_rows)

    con.executemany("""
        INSERT INTO odds_snapshots (snapshot_id, match_id, bookmaker, market, selection, line, odds, taken_at, kind, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, odds_rows)

    con.executemany("""
        INSERT INTO match_weather (match_id, temp_c, precip_mm, wind_kmh)
        VALUES (?, ?, ?, ?)
    """, weather_rows)

    con.commit()
    con.close()

    # Seed 599 real players and starting 11
    seed_players_database(db_path=str(db_path))
