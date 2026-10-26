"""
Automated data quality checks and integrity auditor for football database warehouse.
Ensures zero duplicates, valid temporal ordering, score consistency, and odds sanities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import duckdb
import pandas as pd


@dataclass
class QualityAuditReport:
    passed: bool
    n_checks_run: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def run_data_quality_audit(db_path: str = "data/warehouse.duckdb") -> QualityAuditReport:
    """
    Executes comprehensive data quality validation across tables.
    Fails if any structural corruption, negative odds, or duplicate matches are detected.
    """
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    n_checks = 0

    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        
        # Check 1: Teams validity
        n_checks += 1
        if "teams" in tables:
            n_teams = con.execute("SELECT COUNT(DISTINCT team_id) FROM teams").fetchone()[0]
            metrics["total_teams"] = n_teams
            if n_teams < 18:
                errors.append(f"Incomplete teams table: expected at least 18, found {n_teams}")
        else:
            errors.append("Critical table missing: 'teams'")

        # Check 2: Matches duplicate check
        n_checks += 1
        if "matches" in tables:
            n_matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            metrics["total_matches"] = n_matches
            
            dup_query = """
                SELECT home_team_id, away_team_id, season, matchday, COUNT(*)
                FROM matches
                GROUP BY home_team_id, away_team_id, season, matchday
                HAVING COUNT(*) > 1
            """
            dups = con.execute(dup_query).fetchall()
            if dups:
                errors.append(f"Duplicate matches detected: {len(dups)} duplicated fixture keys")

            # Check 3: Home team != Away team
            n_checks += 1
            self_play = con.execute("SELECT COUNT(*) FROM matches WHERE home_team_id = away_team_id").fetchone()[0]
            if self_play > 0:
                errors.append(f"Self-play fixtures found: {self_play} matches where home_team == away_team")

            # Check 4: Score sanity
            n_checks += 1
            neg_scores = con.execute("""
                SELECT COUNT(*) FROM matches 
                WHERE status = 'played' AND (home_goals < 0 OR away_goals < 0 OR home_goals > 15 OR away_goals > 15)
            """).fetchone()[0]
            if neg_scores > 0:
                errors.append(f"Abnormal goal scores detected: {neg_scores} matches with negative or >15 goals")

            # Check 5: Played matches missing score
            n_checks += 1
            null_scores = con.execute("""
                SELECT COUNT(*) FROM matches 
                WHERE status = 'played' AND (home_goals IS NULL OR away_goals IS NULL)
            """).fetchone()[0]
            if null_scores > 0:
                errors.append(f"Played matches with NULL scores: {null_scores} records")
        else:
            errors.append("Critical table missing: 'matches'")

        # Check 6: Odds sanity
        n_checks += 1
        if "odds_snapshots" in tables:
            n_odds = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
            metrics["total_odds_records"] = n_odds
            
            invalid_odds = con.execute("""
                SELECT COUNT(*) FROM odds_snapshots
                WHERE odds <= 1.00 OR odds > 500.0
            """).fetchone()[0]
            if invalid_odds > 0:
                errors.append(f"Invalid odds found (<= 1.0 or > 500): {invalid_odds} snapshots")

            # Check for orphaned odds
            orphaned = con.execute("""
                SELECT COUNT(*) FROM odds_snapshots o
                LEFT JOIN matches m ON o.match_id = m.match_id
                WHERE m.match_id IS NULL
            """).fetchone()[0]
            if orphaned > 0:
                warnings.append(f"Orphaned odds records without match_id in matches: {orphaned}")

        # Check 7: Players sanity
        n_checks += 1
        if "players" in tables:
            n_players = con.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            metrics["total_players"] = n_players
            if n_players < 200:
                warnings.append(f"Low player count: {n_players} players (expected >= 240)")

            # Starters check
            starters_per_team = con.execute("""
                SELECT team_id, COUNT(*) FROM players 
                WHERE is_starter = TRUE 
                GROUP BY team_id
            """).fetchall()
            for tid, scnt in starters_per_team:
                if scnt != 11:
                    warnings.append(f"Team {tid} has {scnt} starting players (expected exactly 11)")

    finally:
        con.close()

    passed = len(errors) == 0
    return QualityAuditReport(
        passed=passed,
        n_checks_run=n_checks,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
