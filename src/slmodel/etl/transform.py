"""High-performance ETL transformation module to parse raw football-data CSVs into DuckDB."""

import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd
import yaml

from slmodel.data.seed_players import seed_players_database

SCHEMA_PATH = Path("src/slmodel/etl/schema.sql")


def parse_football_date(date_str: str, time_str: Optional[str] = None) -> datetime:
    """Parse date and optional time from football-data into UTC datetime."""
    date_str = str(date_str).strip()
    dt = None
    for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(f"Unable to parse date: {date_str}")

    if time_str and pd.notna(time_str) and str(time_str).strip():
        t_str = str(time_str).strip()
        for t_fmt in ["%H:%M", "%H:%M:%S"]:
            try:
                t = datetime.strptime(t_str, t_fmt).time()
                dt = datetime.combine(dt.date(), t)
                break
            except ValueError:
                pass

    return dt.replace(tzinfo=timezone.utc)


class WarehouseETL:
    """Manages populating and updating the DuckDB database warehouse with batch execution."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self.init_schema()
        self.alias_cache: dict[str, int] = {}
        self.max_team_id: int = 0
        self.derby_pairs: set[tuple[int, int]] = set()
        self.crowd_rules: list[tuple[datetime, datetime, str]] = []
        self.known_stadiums: set[int] = set()
        self.load_alias_cache()
        self.load_metadata_configs()

    def init_schema(self) -> None:
        """Create tables if they do not exist."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self.con.execute(sql)

    def load_metadata_configs(
        self,
        stadiums_csv: str = "config/stadiums.csv",
        crowd_csv: str = "config/crowd_overrides.csv",
        derbies_yaml: str = "config/derbies.yaml",
    ) -> None:
        """Load stadiums, crowd overrides, and derby pairs from config files."""
        s_path = Path(stadiums_csv)
        if s_path.exists():
            df_s = pd.read_csv(s_path)
            self.con.execute("DELETE FROM stadiums")
            df_s_db = df_s[["team_id", "stadium_name", "city", "lat", "lon", "altitude_m"]].rename(
                columns={"team_id": "stadium_id", "stadium_name": "name"}
            )
            self.con.register("tmp_stadiums", df_s_db)
            self.con.execute("INSERT INTO stadiums SELECT * FROM tmp_stadiums")
            self.con.unregister("tmp_stadiums")
            self.known_stadiums = set(df_s_db["stadium_id"].astype(int).tolist())

        c_path = Path(crowd_csv)
        if c_path.exists():
            df_c = pd.read_csv(c_path)
            for _, r in df_c.iterrows():
                d_from = datetime.strptime(str(r["date_from"]).strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                d_to = datetime.strptime(str(r["date_to"]).strip(), "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, tzinfo=timezone.utc
                )
                self.crowd_rules.append((d_from, d_to, str(r["crowd_status"]).strip()))

        d_path = Path(derbies_yaml)
        if d_path.exists():
            with open(d_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for item in data.get("derbies", []):
                    t = item.get("teams", [])
                    if len(t) == 2:
                        self.derby_pairs.add((int(t[0]), int(t[1])))
                        self.derby_pairs.add((int(t[1]), int(t[0])))

    def get_crowd_status(self, dt: datetime) -> str:
        """Resolve crowd attendance status ('full', 'closed', 'limited') for a kickoff date."""
        for d_from, d_to, status in self.crowd_rules:
            if d_from <= dt <= d_to:
                return status
        return "full"

    def load_alias_cache(self, alias_csv: str = "config/team_aliases.csv") -> None:
        """Pre-populate in-memory team alias cache from config and database."""
        path = Path(alias_csv)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                teams_to_insert = {}
                aliases_to_insert = []
                for row in reader:
                    tid = int(row["team_id"])
                    cname = row["canonical_name"].strip()
                    sname = row["source_name"].strip()
                    src = row["source"].strip()
                    teams_to_insert[tid] = cname
                    aliases_to_insert.append((src, sname, tid))
                    self.alias_cache[f"{src}:{sname.lower()}"] = tid
                    self.alias_cache[f"team:{cname.lower()}"] = tid
                    if tid > self.max_team_id:
                        self.max_team_id = tid

            for tid, cname in teams_to_insert.items():
                self.con.execute(
                    "INSERT OR IGNORE INTO teams (team_id, name) VALUES (?, ?)",
                    [tid, cname],
                )
            for src, sname, tid in aliases_to_insert:
                self.con.execute(
                    "INSERT OR REPLACE INTO team_aliases (source, source_name, team_id) VALUES (?, ?, ?)",
                    [src, sname, tid],
                )

        rows = self.con.execute("SELECT source, source_name, team_id FROM team_aliases").fetchall()
        for src, sname, tid in rows:
            self.alias_cache[f"{src}:{sname.lower()}"] = tid
            if tid > self.max_team_id:
                self.max_team_id = tid

        t_rows = self.con.execute("SELECT team_id, name FROM teams").fetchall()
        for tid, name in t_rows:
            self.alias_cache[f"team:{name.lower()}"] = tid
            if tid > self.max_team_id:
                self.max_team_id = tid

    def resolve_team(self, source_name: str, source: str = "football-data") -> int:
        """Fast in-memory team resolution."""
        sname = str(source_name).strip()
        key = f"{source}:{sname.lower()}"
        if key in self.alias_cache:
            return self.alias_cache[key]

        team_key = f"team:{sname.lower()}"
        if team_key in self.alias_cache:
            return self.alias_cache[team_key]

        self.max_team_id += 1
        new_id = self.max_team_id
        self.con.execute("INSERT INTO teams (team_id, name) VALUES (?, ?)", [new_id, sname])
        self.con.execute(
            "INSERT INTO team_aliases (source, source_name, team_id) VALUES (?, ?, ?)",
            [source, sname, new_id],
        )
        self.alias_cache[key] = new_id
        self.alias_cache[team_key] = new_id
        return new_id

    def is_forfeit(self, season: str, dt: datetime, home_name: str, away_name: str) -> bool:
        """Check if match falls under 2023 earthquake forfeit rules."""
        if season == "2223" and dt >= datetime(2023, 2, 6, tzinfo=timezone.utc):
            for t in ["hatayspor", "gaziantep"]:
                if t in home_name.lower() or t in away_name.lower():
                    return True
        return False

    def transform_all_cached_seasons(self, raw_dir: str = "data/raw/footballdata") -> dict[str, int]:
        """Ingest all cached season CSVs into DuckDB warehouse using high-speed DataFrame bulk load."""
        raw_path = Path(raw_dir)
        files = sorted(list(raw_path.glob("T1_*.csv")))
        stats = {"seasons": len(files), "matches": 0, "odds": 0}

        # Preserve prediction_log when refreshing match data, clean child and team tables first
        self.con.execute("DELETE FROM odds_snapshots")
        self.con.execute("DELETE FROM team_match_stats")

        existing_tables = [t[0] for t in self.con.execute("SHOW TABLES").fetchall()]
        if "player_match_stats" in existing_tables:
            self.con.execute("DELETE FROM player_match_stats")
        if "player_season_stats" in existing_tables:
            self.con.execute("DELETE FROM player_season_stats")
        if "team_synergy_units" in existing_tables:
            self.con.execute("DELETE FROM team_synergy_units")
        if "players" in existing_tables:
            self.con.execute("DELETE FROM players")

        self.con.execute("CHECKPOINT")
        self.con.execute("DELETE FROM matches")

        self.con.execute("DELETE FROM team_aliases")
        self.con.execute("CHECKPOINT")
        self.con.execute("DELETE FROM teams")
        self.alias_cache.clear()
        self.max_team_id = 0
        self.load_alias_cache()

        matches_rows = []
        stats_rows = []
        odds_rows = []

        match_id_counter = 0
        snapshot_id_counter = 0

        for f in files:
            season = f.stem.replace("T1_", "")
            data_bytes = f.read_bytes()
            src_hash = hashlib.sha256(data_bytes).hexdigest()[:16]

            df = None
            for enc in ["utf-8-sig", "windows-1252", "iso-8859-9", "utf-8"]:
                try:
                    df = pd.read_csv(f, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                continue

            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
            season_teams = set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique())
            matches_per_round = max(1, len(season_teams) // 2)

            for season_row_idx, (_, row) in enumerate(df.iterrows()):
                try:
                    kickoff = parse_football_date(row["Date"], row.get("Time"))
                except Exception:
                    continue

                home_raw = str(row["HomeTeam"]).strip()
                away_raw = str(row["AwayTeam"]).strip()
                home_id = self.resolve_team(home_raw)
                away_id = self.resolve_team(away_raw)

                status = "played"
                if self.is_forfeit(season, kickoff, home_raw, away_raw):
                    status = "forfeit"

                fthg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None
                ftag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None
                hthg = int(row["HTHG"]) if pd.notna(row.get("HTHG")) else None
                htag = int(row["HTAG"]) if pd.notna(row.get("HTAG")) else None
                referee = str(row["Referee"]).strip() if pd.notna(row.get("Referee")) else None

                match_id_counter += 1
                mid = match_id_counter
                matchday_num = (season_row_idx // matches_per_round) + 1

                crowd_status = self.get_crowd_status(kickoff)
                is_derby_flag = (home_id, away_id) in self.derby_pairs
                stadium_id_val = home_id if home_id in self.known_stadiums else None

                matches_rows.append({
                    "match_id": mid,
                    "league": "T1",
                    "season": season,
                    "matchday": matchday_num,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_goals": fthg,
                    "away_goals": ftag,
                    "ht_home_goals": hthg,
                    "ht_away_goals": htag,
                    "referee": referee,
                    "stadium_id": stadium_id_val,
                    "status": status,
                    "crowd": crowd_status,
                    "is_derby": is_derby_flag,
                    "src_hash": src_hash,
                })

                # Team match stats (including 2026/27 HxG / AxG Expected Goals)
                hs = int(row["HS"]) if pd.notna(row.get("HS")) else None
                as_ = int(row["AS"]) if pd.notna(row.get("AS")) else None
                hst = int(row["HST"]) if pd.notna(row.get("HST")) else None
                ast = int(row["AST"]) if pd.notna(row.get("AST")) else None
                hc = int(row["HC"]) if pd.notna(row.get("HC")) else None
                ac = int(row["AC"]) if pd.notna(row.get("AC")) else None
                hf = int(row["HF"]) if pd.notna(row.get("HF")) else None
                af = int(row["AF"]) if pd.notna(row.get("AF")) else None
                hy = int(row["HY"]) if pd.notna(row.get("HY")) else None
                ay = int(row["AY"]) if pd.notna(row.get("AY")) else None
                hr = int(row["HR"]) if pd.notna(row.get("HR")) else None
                ar = int(row["AR"]) if pd.notna(row.get("AR")) else None
                hxg = float(row["HxG"]) if pd.notna(row.get("HxG")) else None
                axg = float(row["AxG"]) if pd.notna(row.get("AxG")) else None

                if hs is not None or hst is not None or hc is not None or hxg is not None:
                    stats_rows.append({
                        "match_id": mid, "team_id": home_id, "is_home": True,
                        "shots": hs, "shots_on_target": hst, "corners": hc, "fouls": hf,
                        "yellow": hy, "red": hr, "xg": hxg
                    })
                    stats_rows.append({
                        "match_id": mid, "team_id": away_id, "is_home": False,
                        "shots": as_, "shots_on_target": ast, "corners": ac, "fouls": af,
                        "yellow": ay, "red": ar, "xg": axg
                    })

                # Odds
                odds_to_record = []
                if pd.notna(row.get("B365H")) and pd.notna(row.get("B365D")) and pd.notna(row.get("B365A")):
                    odds_to_record.extend([
                        ("Bet365", "1x2", "1", None, float(row["B365H"]), "file_main"),
                        ("Bet365", "1x2", "X", None, float(row["B365D"]), "file_main"),
                        ("Bet365", "1x2", "2", None, float(row["B365A"]), "file_main"),
                    ])
                if pd.notna(row.get("B365CH")) and pd.notna(row.get("B365CD")) and pd.notna(row.get("B365CA")):
                    odds_to_record.extend([
                        ("Bet365", "1x2", "1", None, float(row["B365CH"]), "closing"),
                        ("Bet365", "1x2", "X", None, float(row["B365CD"]), "closing"),
                        ("Bet365", "1x2", "2", None, float(row["B365CA"]), "closing"),
                    ])
                if pd.notna(row.get("PSCH")) and pd.notna(row.get("PSCD")) and pd.notna(row.get("PSCA")):
                    odds_to_record.extend([
                        ("Pinnacle", "1x2", "1", None, float(row["PSCH"]), "closing"),
                        ("Pinnacle", "1x2", "X", None, float(row["PSCD"]), "closing"),
                        ("Pinnacle", "1x2", "2", None, float(row["PSCA"]), "closing"),
                    ])
                if pd.notna(row.get("PSH")) and pd.notna(row.get("PSD")) and pd.notna(row.get("PSA")):
                    odds_to_record.extend([
                        ("Pinnacle", "1x2", "1", None, float(row["PSH"]), "open"),
                        ("Pinnacle", "1x2", "X", None, float(row["PSD"]), "open"),
                        ("Pinnacle", "1x2", "2", None, float(row["PSA"]), "open"),
                    ])
                if pd.notna(row.get("AvgCH")) and pd.notna(row.get("AvgCD")) and pd.notna(row.get("AvgCA")):
                    odds_to_record.extend([
                        ("MarketAvg", "1x2", "1", None, float(row["AvgCH"]), "closing"),
                        ("MarketAvg", "1x2", "X", None, float(row["AvgCD"]), "closing"),
                        ("MarketAvg", "1x2", "2", None, float(row["AvgCA"]), "closing"),
                    ])
                for prefix, bkmk in [("AvgC", "MarketAvg"), ("B365C", "Bet365"), ("PC", "Pinnacle")]:
                    o_col = f"{prefix}>2.5"
                    u_col = f"{prefix}<2.5"
                    if pd.notna(row.get(o_col)) and pd.notna(row.get(u_col)):
                        odds_to_record.extend([
                            (bkmk, "ou", "over", 2.5, float(row[o_col]), "closing"),
                            (bkmk, "ou", "under", 2.5, float(row[u_col]), "closing"),
                        ])
                if pd.notna(row.get("AHCh")) and pd.notna(row.get("AvgCAHH")) and pd.notna(row.get("AvgCAHA")):
                    ah_line = float(row["AHCh"])
                    odds_to_record.extend([
                        ("MarketAvg", "ah", "home", ah_line, float(row["AvgCAHH"]), "closing"),
                        ("MarketAvg", "ah", "away", -ah_line, float(row["AvgCAHA"]), "closing"),
                    ])

                for bkmk, mkt, sel, line, price, kind in odds_to_record:
                    if price > 1.0:
                        snapshot_id_counter += 1
                        odds_rows.append({
                            "snapshot_id": snapshot_id_counter,
                            "match_id": mid,
                            "bookmaker": bkmk,
                            "market": mkt,
                            "selection": sel,
                            "line": line,
                            "odds": price,
                            "taken_at": None,
                            "kind": kind,
                            "source": "football-data",
                        })

        # High speed bulk insert via DataFrames
        df_matches = pd.DataFrame(matches_rows)
        df_stats = pd.DataFrame(stats_rows)
        df_odds = pd.DataFrame(odds_rows)

        if not df_matches.empty:
            self.con.register("tmp_matches", df_matches)
            self.con.execute("INSERT INTO matches SELECT * FROM tmp_matches")
            self.con.unregister("tmp_matches")

        if not df_stats.empty:
            self.con.register("tmp_stats", df_stats)
            self.con.execute("INSERT INTO team_match_stats SELECT * FROM tmp_stats")
            self.con.unregister("tmp_stats")

        if not df_odds.empty:
            self.con.register("tmp_odds", df_odds)
            self.con.execute("INSERT INTO odds_snapshots SELECT * FROM tmp_odds")
            self.con.unregister("tmp_odds")

        self.con.commit()

        # Re-populate players and synergy tables
        try:
            seed_players_database(db_path=str(self.db_path))
        except Exception:
            pass

        stats["matches"] = len(matches_rows)
        stats["odds"] = len(odds_rows)
        return stats

    def close(self) -> None:
        self.con.close()
