"""Contextual features: saturating rest, match congestion, distance, altitude shock, and derbies."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd
import yaml

from slmodel.config import get_model_config


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth (in km)."""
    r = 6371.0  # Earth's radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


def saturating_rest(rest_days: float) -> float:
    """
    Saturating rest response function:
    f(d) = 1.0 - exp(-max(0, d - 2) / 2.5)
    
    Captures non-linear fatigue recovery: going from 2 to 4 days of rest is huge,
    going from 7 to 10 days produces negligible incremental benefit.
    """
    excess_days = max(0.0, float(rest_days) - 2.0)
    return float(1.0 - np.exp(-excess_days / 2.5))


class ContextFeatureExtractor:
    """Extracts orthogonal context and schedule features strictly prior to kickoff_utc."""

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        derbies_path: str = "config/derbies.yaml",
    ) -> None:
        self.db_path = db_path
        self.derbies_path = derbies_path
        self.cfg = get_model_config()
        self._referee_engine = None
        self._load_derbies()
        self._load_stadiums()
        self._load_external_fixtures()

    def _load_derbies(self) -> None:
        p = Path(self.derbies_path)
        self.derby_pairs: set[tuple[int, int]] = set()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                for item in data.get("derbies", []):
                    t = item.get("teams", [])
                    if len(t) == 2:
                        self.derby_pairs.add((int(t[0]), int(t[1])))
                        self.derby_pairs.add((int(t[1]), int(t[0])))

    def _load_stadiums(self) -> None:
        con = duckdb.connect(self.db_path)
        try:
            df = con.execute("SELECT stadium_id, lat, lon, altitude_m FROM stadiums").fetchdf()
            self.stadiums = df.set_index("stadium_id").to_dict(orient="index")
        finally:
            con.close()

    def _load_external_fixtures(self) -> None:
        """Load external European & Cup fixtures for unified calendar fatigue analysis."""
        self.external_fixtures_df: pd.DataFrame = pd.DataFrame()
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "external_fixtures" in tables:
                self.external_fixtures_df = con.execute("""
                    SELECT team_id, kickoff_utc, competition, is_home, status
                    FROM external_fixtures
                """).fetchdf()
                if not self.external_fixtures_df.empty:
                    k_s = pd.to_datetime(self.external_fixtures_df["kickoff_utc"])
                    if hasattr(k_s.dt, "tz") and k_s.dt.tz is not None:
                        self.external_fixtures_df["kickoff_utc"] = k_s.dt.tz_convert("UTC").dt.tz_localize(None)
                    else:
                        self.external_fixtures_df["kickoff_utc"] = k_s
            con.close()
        except Exception:
            self.external_fixtures_df = pd.DataFrame()

    def is_derby(self, home_team_id: int, away_team_id: int) -> bool:
        return (home_team_id, away_team_id) in self.derby_pairs

    def extract_features_for_match(
        self,
        home_team_id: int,
        away_team_id: int,
        kickoff_utc: datetime,
        matches_history: pd.DataFrame,
        referee_name: Optional[str] = None,
        match_id: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Extract strictly as-of context features given prior match history.
        
        Args:
            home_team_id: Home club ID.
            away_team_id: Away club ID.
            kickoff_utc: Cutoff timestamp.
            matches_history: Prior matches with status='played' and kickoff_utc < target kickoff.
            referee_name: Optional officiating referee name for flow and tempo adjustments.
            
        Returns:
            dict containing orthogonal context deviations.
        """
        target_ts = pd.to_datetime(kickoff_utc)
        if target_ts.tz is not None:
            target_ts = target_ts.tz_convert("UTC").tz_localize(None)

        prior_history = matches_history.copy()
        k_series = pd.to_datetime(prior_history["kickoff_utc"])
        if hasattr(k_series.dt, "tz") and k_series.dt.tz is not None:
            prior_history["kickoff_utc"] = k_series.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            prior_history["kickoff_utc"] = k_series

        prior = prior_history[prior_history["kickoff_utc"] < target_ts]

        # 1. Rest days and match congestion across unified calendar (league + European / Cup)
        def get_last_any_match_info(team_id: int) -> tuple[Optional[pd.Timestamp], bool, int]:
            """Returns (last_kickoff, is_recent_away_euro, n_14d_matches)."""
            all_dates: list[tuple[pd.Timestamp, bool]] = []
            team_prior = prior[(prior["home_team_id"] == team_id) | (prior["away_team_id"] == team_id)]
            for _, r in team_prior.iterrows():
                all_dates.append((pd.to_datetime(r["kickoff_utc"]), False))

            if not self.external_fixtures_df.empty:
                ext_prior = self.external_fixtures_df[
                    (self.external_fixtures_df["team_id"] == team_id) &
                    (self.external_fixtures_df["kickoff_utc"] < target_ts)
                ]
                for _, r in ext_prior.iterrows():
                    is_away_euro = (not bool(r.get("is_home", True))) and ("UEFA" in str(r.get("competition", "")))
                    all_dates.append((pd.to_datetime(r["kickoff_utc"]), is_away_euro))

            if not all_dates:
                return None, False, 0

            all_dates.sort(key=lambda x: x[0])
            last_dt, last_away_euro = all_dates[-1]
            diff_days = (target_ts - last_dt).total_seconds() / 86400.0
            is_recent_away_euro = bool(last_away_euro and diff_days <= 4.0)

            w_start = target_ts - pd.Timedelta(days=14)
            n_14d = sum(1 for d, _ in all_dates if d >= w_start)

            return last_dt, is_recent_away_euro, n_14d

        h_last, h_away_euro_recent, h_cong = get_last_any_match_info(home_team_id)
        a_last, a_away_euro_recent, a_cong = get_last_any_match_info(away_team_id)

        h_rest = (target_ts - h_last).total_seconds() / 86400.0 if h_last is not None else 7.0
        a_rest = (target_ts - a_last).total_seconds() / 86400.0 if a_last is not None else 7.0

        # Bound rest days to realistic season limits
        h_rest = float(np.clip(h_rest, 2.0, 21.0))
        a_rest = float(np.clip(a_rest, 2.0, 21.0))

        h_sat_rest = saturating_rest(h_rest)
        a_sat_rest = saturating_rest(a_rest)
        net_rest_advantage = h_sat_rest - a_sat_rest
        net_congestion = float(h_cong - a_cong)

        # European fatigue penalty (only active if euro_fatigue_enabled in config)
        euro_fatigue_penalty_h = -0.025 if (self.cfg.external_factors.euro_fatigue_enabled and h_away_euro_recent) else 0.0
        euro_fatigue_penalty_a = -0.025 if (self.cfg.external_factors.euro_fatigue_enabled and a_away_euro_recent) else 0.0

        # 3. Geography & Stadium Shock
        h_stad = self.stadiums.get(home_team_id, {"lat": 41.0, "lon": 29.0, "altitude_m": 50.0})
        a_stad = self.stadiums.get(away_team_id, {"lat": 41.0, "lon": 29.0, "altitude_m": 50.0})

        distance_km = haversine_km(h_stad["lat"], h_stad["lon"], a_stad["lat"], a_stad["lon"])
        altitude_diff_m = float(h_stad["altitude_m"] - a_stad["altitude_m"])  # positive means playing at higher altitude than away team's base

        # 4. Derby
        derby_flag = 1.0 if self.is_derby(home_team_id, away_team_id) else 0.0

        # 5. Form Momentum (last 5 matches net xG superiority vs long-term baseline)
        def _compute_team_momentum(team_id: int) -> float:
            tm = prior[(prior["home_team_id"] == team_id) | (prior["away_team_id"] == team_id)]
            if len(tm) < 2:
                return 0.0

            has_hxg = "hxg" in tm.columns and tm["hxg"].notna().any()
            diffs: list[float] = []
            for _, r in tm.tail(5).iterrows():
                if r["home_team_id"] == team_id:
                    xf = r["hxg"] if has_hxg and pd.notna(r["hxg"]) else (r.get("hs", 0) * 0.1 if pd.notna(r.get("hs")) else r["home_goals"])
                    xa = r["axg"] if has_hxg and pd.notna(r["axg"]) else (r.get("as_", 0) * 0.1 if pd.notna(r.get("as_")) else r["away_goals"])
                else:
                    xf = r["axg"] if has_hxg and pd.notna(r["axg"]) else (r.get("as_", 0) * 0.1 if pd.notna(r.get("as_")) else r["away_goals"])
                    xa = r["hxg"] if has_hxg and pd.notna(r["hxg"]) else (r.get("hs", 0) * 0.1 if pd.notna(r.get("hs")) else r["home_goals"])
                diffs.append(float(xf - xa))

            if not diffs:
                return 0.0
            w = [0.85 ** j for j in range(len(diffs))][::-1]
            recent_xgd = float(np.average(diffs, weights=w))

            all_diffs: list[float] = []
            for _, r in tm.tail(30).iterrows():
                if r["home_team_id"] == team_id:
                    xf = r["hxg"] if has_hxg and pd.notna(r["hxg"]) else (r.get("hs", 0) * 0.1 if pd.notna(r.get("hs")) else r["home_goals"])
                    xa = r["axg"] if has_hxg and pd.notna(r["axg"]) else (r.get("as_", 0) * 0.1 if pd.notna(r.get("as_")) else r["away_goals"])
                else:
                    xf = r["axg"] if has_hxg and pd.notna(r["axg"]) else (r.get("as_", 0) * 0.1 if pd.notna(r.get("as_")) else r["away_goals"])
                    xa = r["hxg"] if has_hxg and pd.notna(r["hxg"]) else (r.get("hs", 0) * 0.1 if pd.notna(r.get("hs")) else r["home_goals"])
                all_diffs.append(float(xf - xa))
            long_xgd = float(np.mean(all_diffs)) if all_diffs else 0.0
            return float(recent_xgd - long_xgd)

        form_h = _compute_team_momentum(home_team_id)
        form_a = _compute_team_momentum(away_team_id)

        # 5b. Decomposed Attacking Form, Defensive Vulnerability, Venue Edge & In-Season Exposure
        def _compute_team_form_profile(team_id: int, is_home_target: bool) -> dict[str, float]:
            tm = prior[(prior["home_team_id"] == team_id) | (prior["away_team_id"] == team_id)]
            if len(tm) == 0 or "home_goals" not in tm.columns or "away_goals" not in tm.columns:
                return {"att_form": 0.0, "def_vuln": 0.0, "venue_edge": 0.0, "in_season_matches": 0.0}

            has_hxg = "hxg" in tm.columns and tm["hxg"].notna().any()
            rec = tm.tail(6)
            long_tm = tm.tail(30)

            def _extract_series(sub_df: pd.DataFrame) -> tuple[list[float], list[float]]:
                f_vals: list[float] = []
                a_vals: list[float] = []
                for _, r in sub_df.iterrows():
                    if r["home_team_id"] == team_id:
                        gf = float(r["home_goals"])
                        ga = float(r["away_goals"])
                        xf = float(r["hxg"]) if has_hxg and pd.notna(r["hxg"]) else gf
                        xa = float(r["axg"]) if has_hxg and pd.notna(r["axg"]) else ga
                    else:
                        gf = float(r["away_goals"])
                        ga = float(r["home_goals"])
                        xf = float(r["axg"]) if has_hxg and pd.notna(r["axg"]) else gf
                        xa = float(r["hxg"]) if has_hxg and pd.notna(r["hxg"]) else ga
                    f_vals.append(0.60 * xf + 0.40 * gf)
                    a_vals.append(0.60 * xa + 0.40 * ga)
                return f_vals, a_vals

            r_att, r_def = _extract_series(rec)
            l_att, l_def = _extract_series(long_tm)

            w = [0.80 ** j for j in range(len(r_att))][::-1]
            rec_att_mean = float(np.average(r_att, weights=w))
            rec_def_mean = float(np.average(r_def, weights=w))

            base_att = float(np.mean(l_att)) if len(l_att) >= 8 else 1.35
            base_def = float(np.mean(l_def)) if len(l_def) >= 8 else 1.35

            att_form = float(rec_att_mean - base_att)
            def_vuln = float(rec_def_mean - base_def)

            v_tm = tm[tm["home_team_id"] == team_id].tail(6) if is_home_target else tm[tm["away_team_id"] == team_id].tail(6)
            if len(v_tm) >= 2:
                v_diffs: list[float] = []
                for _, r in v_tm.iterrows():
                    if is_home_target:
                        gf, ga = float(r["home_goals"]), float(r["away_goals"])
                        xf = float(r["hxg"]) if has_hxg and pd.notna(r["hxg"]) else gf
                        xa = float(r["axg"]) if has_hxg and pd.notna(r["axg"]) else ga
                    else:
                        gf, ga = float(r["away_goals"]), float(r["home_goals"])
                        xf = float(r["axg"]) if has_hxg and pd.notna(r["axg"]) else gf
                        xa = float(r["hxg"]) if has_hxg and pd.notna(r["hxg"]) else ga
                    v_diffs.append(0.5 * (gf - ga) + 0.5 * (xf - xa))
                venue_edge = float(np.mean(v_diffs) - (0.25 if is_home_target else -0.25))
            else:
                venue_edge = 0.0

            if "season" in tm.columns and len(prior) > 0:
                cur_season = str(prior["season"].iloc[-1])
                n_in_season = float(len(tm[tm["season"].astype(str) == cur_season]))
            else:
                n_in_season = float(min(len(tm), 10))

            return {
                "att_form": att_form,
                "def_vuln": def_vuln,
                "venue_edge": venue_edge,
                "in_season_matches": n_in_season,
            }

        prof_h = _compute_team_form_profile(home_team_id, is_home_target=True)
        prof_a = _compute_team_form_profile(away_team_id, is_home_target=False)

        # 6. Referee Flow & Tempo Adjustments
        ref_delta_h = 0.0
        ref_delta_a = 0.0
        ref_draw_inf = 0.0
        ref_whistle = 0.0
        ref_card_mult = 1.0

        if referee_name and str(referee_name).strip() != "":
            if self._referee_engine is None:
                from slmodel.models.referee_engine import RefereeEngine
                self._referee_engine = RefereeEngine(db_path=self.db_path)
            
            prof = self._referee_engine.get_referee_profile(
                referee_name, as_of=kickoff_utc, matches_history=matches_history
            )
            flow = self._referee_engine.compute_flow_and_tempo_adjustments(prof)
            ref_delta_h = flow["delta_h"]
            ref_delta_a = flow["delta_a"]
            ref_draw_inf = flow["draw_inflation"]
            ref_whistle = flow["whistle_tempo_index"]
            ref_card_mult = float(prof.get("card_strictness_mult", 1.0))

        # 7. Weather Conditions (from match_weather)
        weather_delta_pace = 0.0
        weather_temp = 18.0
        weather_precip = 0.0
        weather_wind = 12.0
        if match_id is not None:
            try:
                con = duckdb.connect(self.db_path, read_only=True)
                w_row = con.execute("SELECT temp_c, precip_mm, wind_kmh FROM match_weather WHERE match_id = ?", [match_id]).fetchone()
                con.close()
                if w_row:
                    weather_temp = float(w_row[0]) if w_row[0] is not None else 18.0
                    weather_precip = float(w_row[1]) if w_row[1] is not None else 0.0
                    weather_wind = float(w_row[2]) if w_row[2] is not None else 12.0

                    if self.cfg.external_factors.weather_enabled:
                        if weather_precip > 2.0:
                            weather_delta_pace -= 0.025
                        if weather_wind > 35.0:
                            weather_delta_pace -= 0.020
                        if weather_temp < 2.0 or weather_temp > 32.0:
                            weather_delta_pace -= 0.015
            except Exception:
                pass

        return {
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "net_rest_advantage": net_rest_advantage,
            "home_14d_matches": float(h_cong),
            "away_14d_matches": float(a_cong),
            "net_congestion": net_congestion,
            "euro_fatigue_h": euro_fatigue_penalty_h,
            "euro_fatigue_a": euro_fatigue_penalty_a,
            "weather_temp": weather_temp,
            "weather_precip": weather_precip,
            "weather_wind": weather_wind,
            "weather_delta_pace": weather_delta_pace,
            "travel_distance_km": distance_km,
            "altitude_diff_m": altitude_diff_m,
            "is_derby": derby_flag,
            "form_momentum_h": form_h,
            "form_momentum_a": form_a,
            "form_att_h": prof_h["att_form"],
            "form_def_vuln_h": prof_h["def_vuln"],
            "venue_edge_h": prof_h["venue_edge"],
            "in_season_matches_h": prof_h["in_season_matches"],
            "form_att_a": prof_a["att_form"],
            "form_def_vuln_a": prof_a["def_vuln"],
            "venue_edge_a": prof_a["venue_edge"],
            "in_season_matches_a": prof_a["in_season_matches"],
            "ref_delta_h": ref_delta_h,
            "ref_delta_a": ref_delta_a,
            "ref_draw_inf": ref_draw_inf,
            "ref_whistle_index": ref_whistle,
            "ref_card_mult": ref_card_mult,
        }
