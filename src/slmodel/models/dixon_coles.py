"""Dixon-Coles bivariate Poisson model with analytical gradient, Hessian Laplace covariance, shot-quality composite likelihood, and Empirical Bayes priors."""

from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from slmodel.simulate.score_matrix import (
    score_matrix,
    posterior_predictive_score_matrix,
)


def compute_squad_priors(db_path: str = "data/warehouse.duckdb") -> dict[int, tuple[float, float]]:
    """
    Compute Empirical Bayes attack/defense prior shifts from Mackolik squad valuations
    and starter ratings stored in the DuckDB warehouse.
    Returns dict mapping team_id -> (att_prior, def_prior).
    """
    try:
        import duckdb
        con = duckdb.connect(db_path)
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if "players" not in tables:
            con.close()
            return {}
        p_df = con.execute("""
            SELECT p.team_id, 
                   SUM(p.market_value_eur) as squad_val,
                   AVG(CASE WHEN p.is_starter THEN p.overall_rating ELSE NULL END) as starter_ovr,
                   AVG(CASE WHEN p.is_starter AND p.position IN ('FW', 'AM', 'RW', 'LW', 'ST') THEN p.att_rating ELSE NULL END) as starter_att,
                   AVG(CASE WHEN p.is_starter AND p.position IN ('CB', 'LB', 'RB', 'GK', 'DM') THEN p.def_rating ELSE NULL END) as starter_def
            FROM players p
            GROUP BY p.team_id
        """).fetchdf()
        con.close()
        if p_df.empty or len(p_df) < 5:
            return {}

        log_v = np.log(np.clip(p_df["squad_val"].to_numpy(dtype=float), 1e6, None))
        z_v = (log_v - np.mean(log_v)) / (np.std(log_v) + 1e-6)

        ovr = p_df["starter_ovr"].fillna(np.mean(p_df["starter_ovr"].dropna())).to_numpy(dtype=float)
        z_ovr = (ovr - np.mean(ovr)) / (np.std(ovr) + 1e-6)

        att = p_df["starter_att"].fillna(np.mean(p_df["starter_att"].dropna())).to_numpy(dtype=float)
        z_att = (att - np.mean(att)) / (np.std(att) + 1e-6)

        dfn = p_df["starter_def"].fillna(np.mean(p_df["starter_def"].dropna())).to_numpy(dtype=float)
        z_def = (dfn - np.mean(dfn)) / (np.std(dfn) + 1e-6)

        squad_priors = {}
        for i, tid in enumerate(p_df["team_id"]):
            att_score = float(0.45 * z_v[i] + 0.30 * z_ovr[i] + 0.25 * z_att[i])
            def_score = float(0.45 * z_v[i] + 0.30 * z_ovr[i] + 0.25 * z_def[i])
            squad_priors[int(tid)] = (att_score, def_score)
        return squad_priors
    except Exception:
        return {}


class DixonColesModel:
    """
    Estimates team attack/defense ratings, home advantage, closed-door structural shift,
    and low-score dependency rho using time-decayed composite MAP with exact analytical gradients.
    """

    def __init__(
        self,
        half_life_years: float = 1.25,
        ridge_att: float = 1.5,
        ridge_def: float = 1.5,
        shot_alpha: float = 0.40,
        season_carryover: float = 0.85,
        in_season_boost: float = 1.0,
        promoted_prior_att: float = -0.15,
        promoted_prior_def: float = -0.15,
        prior_exposure_scale: float = 12.0,
        squad_prior_weight: float = 0.0,
        early_season_prior_weight: float = 0.0,
        form_momentum_weight: float = 0.0,
        base_draw_inflation: float = 0.0,
        closeness_draw_threshold: float = 0.0,
        closeness_draw_boost: float = 0.0,
        fixed_rho: Optional[float] = None,
        team_squad_priors: Optional[dict[int, tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> None:
        self.half_life_years = half_life_years
        self.ridge_att = ridge_att
        self.ridge_def = ridge_def
        self.shot_alpha = shot_alpha
        self.season_carryover = season_carryover
        self.in_season_boost = in_season_boost
        self.promoted_prior_att = promoted_prior_att
        self.promoted_prior_def = promoted_prior_def
        self.prior_exposure_scale = prior_exposure_scale
        self.squad_prior_weight = squad_prior_weight
        self.early_season_prior_weight = early_season_prior_weight
        self.form_momentum_weight = form_momentum_weight
        self.base_draw_inflation = base_draw_inflation
        self.closeness_draw_threshold = closeness_draw_threshold
        self.closeness_draw_boost = closeness_draw_boost
        self.fixed_rho = fixed_rho
        self.team_squad_priors = team_squad_priors or {}

        # Fitted parameters
        self.team_ids: list[int] = []
        self.team_to_idx: dict[int, int] = {}
        self.att: np.ndarray = np.array([])
        self.def_: np.ndarray = np.array([])
        self.team_exposure: np.ndarray = np.array([])
        self.mu0: float = 0.20
        self.home_adv: float = 0.25
        self.closed_door_delta: float = -0.08
        self.rho: float = -0.05
        self.shot_beta: Optional[np.ndarray] = None
        self.param_cov: Optional[np.ndarray] = None
        self.raw_params: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _unpack(self, params: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Unpack parameter vector applying sum-to-zero constraint."""
        n_free = n_teams - 1
        att_free = params[:n_free]
        def_free = params[n_free: 2 * n_free]

        att = np.append(att_free, -np.sum(att_free))
        def_ = np.append(def_free, -np.sum(def_free))

        mu0 = float(params[-3])
        home_adv = float(params[-2])
        rho = float(0.5 * np.tanh(params[-1]))

        return att, def_, mu0, home_adv, rho

    def fit(
        self,
        df: pd.DataFrame,
        as_of: Optional[datetime] = None,
        sample_weights: Optional[np.ndarray] = None,
        warm_start: bool = False,
    ) -> "DixonColesModel":
        """
        Fit Dixon-Coles model to match history strictly prior to as_of.
        DataFrame must contain: home_team_id, away_team_id, home_goals, away_goals, kickoff_utc, status.
        Optional shot columns (hs, hst, as_, ast, hc, ac) enable composite likelihood.
        """
        valid_mask = (df["status"] == "played") & df["home_goals"].notna() & df["away_goals"].notna()
        valid = df[valid_mask].copy()
        sw_valid = sample_weights[valid_mask.to_numpy()] if (sample_weights is not None and len(sample_weights) == len(df)) else sample_weights

        if as_of is not None:
            as_of_ts = pd.to_datetime(as_of)
            if as_of_ts.tz is not None:
                as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
            k_series = pd.to_datetime(valid["kickoff_utc"])
            if hasattr(k_series.dt, "tz") and k_series.dt.tz is not None:
                k_series = k_series.dt.tz_convert("UTC").dt.tz_localize(None)
            as_of_mask = (k_series < as_of_ts).to_numpy()
            valid = valid[as_of_mask]
            if sw_valid is not None and len(sw_valid) == len(as_of_mask):
                sw_valid = sw_valid[as_of_mask]

        n_matches = len(valid)
        if n_matches < 20:
            raise ValueError(f"Insufficient matches to fit Dixon-Coles: {n_matches}")

        # Collect unique teams
        teams = sorted(list(set(valid["home_team_id"].unique()) | set(valid["away_team_id"].unique())))
        n_teams = len(teams)
        self.team_ids = teams
        self.team_to_idx = {tid: i for i, tid in enumerate(teams)}

        h_idx = np.array([self.team_to_idx[tid] for tid in valid["home_team_id"]], dtype=np.int32)
        a_idx = np.array([self.team_to_idx[tid] for tid in valid["away_team_id"]], dtype=np.int32)
        x_raw = valid["home_goals"].to_numpy(dtype=np.float64)
        y_raw = valid["away_goals"].to_numpy(dtype=np.float64)

        # Time decay weights
        if as_of is None:
            ref_date = valid["kickoff_utc"].max()
        else:
            ref_date = as_of

        dt_series = pd.to_datetime(valid["kickoff_utc"], utc=True)
        ref_dt = pd.to_datetime(ref_date, utc=True)
        age_years = (ref_dt - dt_series).dt.total_seconds().to_numpy() / (365.25 * 86400.0)
        age_years = np.clip(age_years, 0.0, None)

        if self.half_life_years > 0:
            weights = np.power(0.5, age_years / self.half_life_years)
        else:
            weights = np.ones_like(age_years)

        # Off-season squad turnover decay & in-season regime boost
        if (self.season_carryover < 1.0 or self.in_season_boost != 1.0) and "season" in valid.columns and n_matches >= 100:
            seasons_arr = valid["season"].astype(str).to_numpy()
            u_seasons = sorted(list(set(seasons_arr)))
            if len(u_seasons) > 1:
                s_map = {s: idx for idx, s in enumerate(u_seasons)}
                cur_s_idx = s_map[seasons_arr[-1]]
                s_gaps = np.array([max(0, cur_s_idx - s_map[s]) for s in seasons_arr], dtype=np.float64)
                if self.season_carryover < 1.0:
                    weights = weights * np.power(self.season_carryover, s_gaps)
                if self.in_season_boost != 1.0:
                    weights = weights * np.where(s_gaps == 0.0, self.in_season_boost, 1.0)

        if sw_valid is not None and len(sw_valid) == n_matches:
            weights = weights * sw_valid

        # Closed-door indicator
        if "crowd" in valid.columns:
            is_closed = (valid["crowd"] == "closed").to_numpy(dtype=np.float64)
        else:
            is_closed = np.zeros(n_matches, dtype=np.float64)

        # Shot-Quality & xG Hybrid Composite Likelihood (strictly out-of-sample on valid training rows)
        has_shots = all(c in valid.columns for c in ["hs", "hst", "as_", "ast"])
        has_xg = all(c in valid.columns for c in ["hxg", "axg"])
        x_eff = x_raw.copy()
        y_eff = y_raw.copy()

        if self.shot_alpha > 0.0 and has_shots and n_matches >= 60:
            hst_arr = valid["hst"].to_numpy(dtype=np.float64)
            ast_arr = valid["ast"].to_numpy(dtype=np.float64)
            hs_arr = valid["hs"].to_numpy(dtype=np.float64)
            as_arr = valid["as_"].to_numpy(dtype=np.float64)
            hc_arr = valid["hc"].to_numpy(dtype=np.float64) if "hc" in valid.columns else np.zeros_like(hst_arr)
            ac_arr = valid["ac"].to_numpy(dtype=np.float64) if "ac" in valid.columns else np.zeros_like(ast_arr)

            vs = ~np.isnan(hst_arr) & ~np.isnan(ast_arr) & ~np.isnan(hs_arr) & ~np.isnan(as_arr)
            if np.sum(vs) >= 50:
                hst_v = hst_arr[vs]
                ast_v = ast_arr[vs]
                hs_off = np.clip(hs_arr[vs] - hst_v, 0.0, None)
                as_off = np.clip(as_arr[vs] - ast_v, 0.0, None)
                hc_v = np.nan_to_num(hc_arr[vs], nan=0.0)
                ac_v = np.nan_to_num(ac_arr[vs], nan=0.0)

                X_shot = np.vstack([
                    np.column_stack([np.ones_like(hst_v), hst_v, hs_off, hc_v]),
                    np.column_stack([np.ones_like(ast_v), ast_v, as_off, ac_v]),
                ])
                y_shot = np.concatenate([x_raw[vs], y_raw[vs]])
                w_shot = np.concatenate([weights[vs], weights[vs]])
                sw = np.sqrt(w_shot)
                beta_s, _ = nnls(X_shot * sw[:, None], y_shot * sw)
                self.shot_beta = beta_s

                gh = beta_s[0] + beta_s[1] * hst_v + beta_s[2] * hs_off + beta_s[3] * hc_v
                ga = beta_s[0] + beta_s[1] * ast_v + beta_s[2] * as_off + beta_s[3] * ac_v

                x_eff[vs] = (1.0 - self.shot_alpha) * x_raw[vs] + self.shot_alpha * gh
                y_eff[vs] = (1.0 - self.shot_alpha) * y_raw[vs] + self.shot_alpha * ga

        # Incorporate true match xG (e.g. 2026/27 HxG / AxG) where available
        if self.shot_alpha > 0.0 and has_xg:
            hxg_arr = valid["hxg"].to_numpy(dtype=np.float64)
            axg_arr = valid["axg"].to_numpy(dtype=np.float64)
            v_xg = ~np.isnan(hxg_arr) & ~np.isnan(axg_arr)
            if np.any(v_xg):
                # Blend actual goals, shot-conversion estimate, and true match xG
                x_eff[v_xg] = 0.55 * x_raw[v_xg] + 0.15 * x_eff[v_xg] + 0.30 * hxg_arr[v_xg]
                y_eff[v_xg] = 0.55 * y_raw[v_xg] + 0.15 * y_eff[v_xg] + 0.30 * axg_arr[v_xg]

        # Effective team sample weights & Exposure-Aware Empirical Bayes priors
        team_w = np.zeros(n_teams, dtype=np.float64)
        np.add.at(team_w, h_idx, weights)
        np.add.at(team_w, a_idx, weights)
        self.team_exposure = team_w

        att_prior = np.zeros(n_teams, dtype=np.float64)
        def_prior = np.zeros(n_teams, dtype=np.float64)

        if n_matches >= 100:
            low_exp = np.exp(-team_w / max(self.prior_exposure_scale, 1.0))
            if self.team_squad_priors and self.squad_prior_weight > 0.0:
                for tid, idx in self.team_to_idx.items():
                    if tid in self.team_squad_priors:
                        sp_att, sp_def = self.team_squad_priors[tid]
                        # Exposure-weighted blend: low exposure gets full squad prior,
                        # high exposure retains residual anchor for current season caliber
                        eff_w = self.squad_prior_weight * (0.35 + 0.65 * low_exp[idx])
                        att_prior[idx] = eff_w * sp_att
                        def_prior[idx] = eff_w * sp_def
                    else:
                        att_prior[idx] = self.promoted_prior_att * low_exp[idx]
                        def_prior[idx] = self.promoted_prior_def * low_exp[idx]
            elif self.promoted_prior_att != 0.0 or self.promoted_prior_def != 0.0:
                att_prior = self.promoted_prior_att * low_exp
                def_prior = self.promoted_prior_def * low_exp

            # Zero-sum gauge constraint
            att_prior -= np.mean(att_prior)
            def_prior -= np.mean(def_prior)

        # Low-score masks for Dixon-Coles tau
        c00 = (x_raw == 0) & (y_raw == 0)
        c01 = (x_raw == 0) & (y_raw == 1)
        c10 = (x_raw == 1) & (y_raw == 0)
        c11 = (x_raw == 1) & (y_raw == 1)

        n_free = n_teams - 1
        if warm_start and self.raw_params is not None and len(self.raw_params) == 2 * n_free + 3:
            init_params = self.raw_params.copy()
        else:
            init_params = np.zeros(2 * n_free + 3, dtype=np.float64)
            init_params[-3] = 0.20   # mu0
            init_params[-2] = 0.25   # home_adv
            init_params[-1] = -0.10  # rho_raw

        closed_shift = self.closed_door_delta * is_closed

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            att_free = params[:n_free]
            def_free = params[n_free: 2 * n_free]
            att = np.append(att_free, -np.sum(att_free))
            def_ = np.append(def_free, -np.sum(def_free))
            mu0 = params[-3]
            home_adv = params[-2]
            rho_raw = params[-1]

            tanh_r = np.tanh(rho_raw)
            rho = 0.5 * tanh_r
            drho_draw = 0.5 * (1.0 - tanh_r * tanh_r)

            eta_h = mu0 + home_adv + closed_shift + att[h_idx] - def_[a_idx]
            eta_a = mu0 + att[a_idx] - def_[h_idx]
            lam = np.exp(eta_h)
            mu = np.exp(eta_a)

            tau_vals = np.ones_like(lam)
            tau_vals[c00] = 1.0 - lam[c00] * mu[c00] * rho
            tau_vals[c01] = 1.0 + lam[c01] * rho
            tau_vals[c10] = 1.0 + mu[c10] * rho
            tau_vals[c11] = 1.0 - rho

            clipped = tau_vals < 1e-6
            tau_safe = np.where(clipped, 1e-6, tau_vals)

            log_lik = (
                np.log(tau_safe)
                + x_eff * eta_h - lam
                + y_eff * eta_a - mu
            )

            nll = -np.sum(weights * log_lik)
            nll += self.ridge_att * np.sum((att - att_prior) ** 2)
            nll += self.ridge_def * np.sum((def_ - def_prior) ** 2)
            nll += 2.0 * (rho ** 2)

            # Analytical gradients wrt eta_h, eta_a, rho
            dtau_deta_h = np.zeros_like(lam)
            dtau_deta_a = np.zeros_like(lam)
            dtau_drho = np.zeros_like(lam)

            act = ~clipped
            m00, m01, m10, m11 = c00 & act, c01 & act, c10 & act, c11 & act

            dtau_deta_h[m00] = (-lam[m00] * mu[m00] * rho) / tau_safe[m00]
            dtau_deta_a[m00] = dtau_deta_h[m00]
            dtau_drho[m00] = (-lam[m00] * mu[m00]) / tau_safe[m00]

            dtau_deta_h[m01] = (lam[m01] * rho) / tau_safe[m01]
            dtau_drho[m01] = lam[m01] / tau_safe[m01]

            dtau_deta_a[m10] = (mu[m10] * rho) / tau_safe[m10]
            dtau_drho[m10] = mu[m10] / tau_safe[m10]

            dtau_drho[m11] = -1.0 / tau_safe[m11]

            g_eta_h = -weights * (dtau_deta_h + x_eff - lam)
            g_eta_a = -weights * (dtau_deta_a + y_eff - mu)
            g_rho = -np.sum(weights * dtau_drho) + 4.0 * rho

            g_att = np.zeros(n_teams, dtype=np.float64)
            g_def = np.zeros(n_teams, dtype=np.float64)
            np.add.at(g_att, h_idx, g_eta_h)
            np.add.at(g_att, a_idx, g_eta_a)
            np.add.at(g_def, a_idx, -g_eta_h)
            np.add.at(g_def, h_idx, -g_eta_a)

            g_att += 2.0 * self.ridge_att * (att - att_prior)
            g_def += 2.0 * self.ridge_def * (def_ - def_prior)

            g_att_free = g_att[:n_free] - g_att[-1]
            g_def_free = g_def[:n_free] - g_def[-1]

            g_mu0 = float(np.sum(g_eta_h + g_eta_a))
            g_hadv = float(np.sum(g_eta_h))
            g_rho_raw = float(g_rho * drho_draw)

            grad = np.concatenate([g_att_free, g_def_free, [g_mu0, g_hadv, g_rho_raw]])
            return float(nll), grad

        res = minimize(
            loss_and_grad,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-7},
        )

        self.raw_params = res.x.copy()
        att, def_, mu0, home_adv, rho = self._unpack(res.x, n_teams)
        self.att = att
        self.def_ = def_
        self.mu0 = mu0
        self.home_adv = home_adv
        self.rho = rho

        # Compute exact analytical Fisher / Hessian covariance for Laplace Posterior Predictive
        self._compute_laplace_covariance(h_idx, a_idx, weights, is_closed)
        self.is_fitted = True

        return self

    def _compute_laplace_covariance(
        self,
        h_idx: np.ndarray,
        a_idx: np.ndarray,
        weights: np.ndarray,
        is_closed: np.ndarray,
    ) -> None:
        """
        Compute the exact analytical Hessian H of the regularized log-likelihood wrt
        full parameter vector [att_0..att_{N-1}, def_0..def_{N-1}, mu0, home_adv]
        under sum-to-zero constraints and invert for Laplace posterior covariance.
        """
        n_teams = len(self.team_ids)
        dim = 2 * n_teams + 2
        H = np.zeros((dim, dim), dtype=np.float64)

        eta_h = self.mu0 + self.home_adv + self.closed_door_delta * is_closed + self.att[h_idx] - self.def_[a_idx]
        eta_a = self.mu0 + self.att[a_idx] - self.def_[h_idx]
        w_lam = weights * np.exp(eta_h)
        w_mu = weights * np.exp(eta_a)

        # Diagonal & off-diagonal second derivatives from Poisson Fisher information
        # Order: 0..N-1: att, N..2N-1: def, 2N: mu0, 2N+1: home_adv
        np.add.at(H, (h_idx, h_idx), w_lam)
        np.add.at(H, (a_idx, a_idx), w_mu)
        np.add.at(H, (n_teams + a_idx, n_teams + a_idx), w_lam)
        np.add.at(H, (n_teams + h_idx, n_teams + h_idx), w_mu)

        # Cross terms between att_h and def_a (from eta_h = att_h - def_a)
        np.add.at(H, (h_idx, n_teams + a_idx), -w_lam)
        np.add.at(H, (n_teams + a_idx, h_idx), -w_lam)

        # Cross terms between att_a and def_h (from eta_a = att_a - def_h)
        np.add.at(H, (a_idx, n_teams + h_idx), -w_mu)
        np.add.at(H, (n_teams + h_idx, a_idx), -w_mu)

        # Cross terms with mu0 (2N) and home_adv (2N+1)
        idx_mu0 = 2 * n_teams
        idx_hadv = 2 * n_teams + 1
        for k in range(n_teams):
            mask_h = h_idx == k
            mask_a = a_idx == k
            s_wh = float(np.sum(w_lam[mask_h]))
            s_wa = float(np.sum(w_mu[mask_a]))
            s_def_a = float(np.sum(w_lam[mask_a]))
            s_def_h = float(np.sum(w_mu[mask_h]))

            H[k, idx_mu0] = H[idx_mu0, k] = s_wh + s_wa
            H[k, idx_hadv] = H[idx_hadv, k] = s_wh
            H[n_teams + k, idx_mu0] = H[idx_mu0, n_teams + k] = -(s_def_a + s_def_h)
            H[n_teams + k, idx_hadv] = H[idx_hadv, n_teams + k] = -s_def_a

        tot_w_lam = float(np.sum(w_lam))
        tot_w_mu = float(np.sum(w_mu))
        H[idx_mu0, idx_mu0] = tot_w_lam + tot_w_mu + 1.0
        H[idx_hadv, idx_hadv] = tot_w_lam + 1.0
        H[idx_mu0, idx_hadv] = H[idx_hadv, idx_mu0] = tot_w_lam

        # Add L2 ridge priors + sum-to-zero soft gauge fixing
        idx_att = np.arange(n_teams)
        idx_def = np.arange(n_teams, 2 * n_teams)
        H[idx_att, idx_att] += 2.0 * self.ridge_att
        H[idx_def, idx_def] += 2.0 * self.ridge_def
        H[:n_teams, :n_teams] += 10.0
        H[n_teams:2 * n_teams, n_teams:2 * n_teams] += 10.0

        try:
            self.param_cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.param_cov = np.linalg.pinv(H)

    def compute_context_adjustments(self, context_features: Optional[dict[str, float]]) -> tuple[float, float, float]:
        """
        Compute orthogonal zero-centered Layer E log-intensity adjustments (delta_h, delta_a, draw_inflation)
        from context features (saturating rest, congestion, altitude shock, travel distance, derby).
        """
        if not context_features:
            return 0.0, 0.0, float(self.base_draw_inflation)

        delta_h = 0.0
        delta_a = 0.0
        draw_inf = float(self.base_draw_inflation)

        # 1. Saturating rest advantage: f(d_H) - f(d_A) in [-1, 1]
        net_rest = float(context_features.get("net_rest_advantage", 0.0))
        delta_h += 0.045 * net_rest
        delta_a -= 0.045 * net_rest

        # 2. 14-day schedule congestion difference
        net_cong = float(context_features.get("net_congestion", 0.0))
        delta_h -= 0.015 * np.clip(net_cong, -3.0, 3.0)
        delta_a += 0.015 * np.clip(net_cong, -3.0, 3.0)

        # 3. Altitude shock: playing at high altitude (>800m above away team's base)
        alt_diff = float(context_features.get("altitude_diff_m", 0.0))
        if alt_diff > 600.0:
            alt_factor = min((alt_diff - 600.0) / 1000.0, 1.5)
            delta_h += 0.035 * alt_factor
            delta_a -= 0.045 * alt_factor

        # 4. Long travel distance fatigue (>600 km)
        dist_km = float(context_features.get("travel_distance_km", 0.0))
        if dist_km > 600.0:
            dist_factor = min((dist_km - 600.0) / 1000.0, 1.0)
            delta_a -= 0.025 * dist_factor

        # 5. Derby match tension (slight draw compression / tactical caution)
        if float(context_features.get("is_derby", 0.0)) > 0.5:
            draw_inf += 0.04

        # 6. Form Momentum (decomposed attack form, opponent defensive vulnerability, and venue edge)
        if "form_att_h" in context_features and self.form_momentum_weight > 0.0:
            f_att_h = float(context_features.get("form_att_h", 0.0))
            f_def_h = float(context_features.get("form_def_vuln_h", 0.0))
            v_edg_h = float(context_features.get("venue_edge_h", 0.0))
            f_att_a = float(context_features.get("form_att_a", 0.0))
            f_def_a = float(context_features.get("form_def_vuln_a", 0.0))
            v_edg_a = float(context_features.get("venue_edge_a", 0.0))
            delta_h += (
                self.form_momentum_weight * float(np.tanh(f_att_h / 0.8))
                + self.form_momentum_weight * float(np.tanh(f_def_a / 0.8))
                + 0.05 * float(np.tanh(v_edg_h))
            )
            delta_a += (
                self.form_momentum_weight * float(np.tanh(f_att_a / 0.8))
                + self.form_momentum_weight * float(np.tanh(f_def_h / 0.8))
                + 0.05 * float(np.tanh(v_edg_a))
            )
        elif self.form_momentum_weight > 0.0:
            form_h = float(context_features.get("form_momentum_h", 0.0))
            form_a = float(context_features.get("form_momentum_a", 0.0))
            delta_h += self.form_momentum_weight * float(np.tanh(form_h / 1.5))
            delta_a += self.form_momentum_weight * float(np.tanh(form_a / 1.5))

        # 7. Referee Flow, Pace & Draw Inflation
        delta_h += float(context_features.get("ref_delta_h", 0.0))
        delta_a += float(context_features.get("ref_delta_a", 0.0))
        draw_inf += float(context_features.get("ref_draw_inf", 0.0))

        return float(delta_h), float(delta_a), float(draw_inf)

    def predict_lambdas(
        self,
        home_team_id: int,
        away_team_id: int,
        crowd: str = "full",
        context_features: Optional[dict[str, float]] = None,
    ) -> tuple[float, float, float]:
        """Predict (lam_home, lam_away, rho) for given team IDs and optional context."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        h_idx = self.team_to_idx.get(home_team_id)
        a_idx = self.team_to_idx.get(away_team_id)

        unseen_prior_att = self.promoted_prior_att if len(self.team_ids) >= 15 else 0.0
        unseen_prior_def = self.promoted_prior_def if len(self.team_ids) >= 15 else 0.0

        if h_idx is not None:
            att_h = self.att[h_idx]
            def_h = self.def_[h_idx]
        elif home_team_id in self.team_squad_priors and self.squad_prior_weight > 0.0:
            sp_att, sp_def = self.team_squad_priors[home_team_id]
            att_h = self.squad_prior_weight * sp_att
            def_h = self.squad_prior_weight * sp_def
        else:
            att_h = unseen_prior_att
            def_h = unseen_prior_def

        if a_idx is not None:
            att_a = self.att[a_idx]
            def_a = self.def_[a_idx]
        elif away_team_id in self.team_squad_priors and self.squad_prior_weight > 0.0:
            sp_att, sp_def = self.team_squad_priors[away_team_id]
            att_a = self.squad_prior_weight * sp_att
            def_a = self.squad_prior_weight * sp_def
        else:
            att_a = unseen_prior_att
            def_a = unseen_prior_def

        delta_h, delta_a, _ = self.compute_context_adjustments(context_features)

        if self.early_season_prior_weight > 0.0 and self.team_squad_priors and context_features:
            n_in_h = float(context_features.get("in_season_matches_h", 10.0))
            n_in_a = float(context_features.get("in_season_matches_a", 10.0))
            w_sh = float(np.exp(-n_in_h / 5.0) * self.early_season_prior_weight)
            w_sa = float(np.exp(-n_in_a / 5.0) * self.early_season_prior_weight)
            sp_h = self.team_squad_priors.get(home_team_id, (0.0, 0.0))
            sp_a = self.team_squad_priors.get(away_team_id, (0.0, 0.0))
            delta_h += w_sh * sp_h[0] - w_sa * sp_a[1]
            delta_a += w_sa * sp_a[0] - w_sh * sp_h[1]

        crowd_adj = self.closed_door_delta if crowd == "closed" else 0.0

        lam = float(np.exp(self.mu0 + self.home_adv + crowd_adj + att_h - def_a + delta_h))
        mu = float(np.exp(self.mu0 + att_a - def_h + delta_a))
        eff_rho = float(self.fixed_rho) if self.fixed_rho is not None else self.rho

        return lam, mu, eff_rho

    def predict_with_uncertainty(
        self,
        home_team_id: int,
        away_team_id: int,
        crowd: str = "full",
        context_features: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """
        Return point-estimate intensities (lam, mu, rho) AND exact Laplace posterior
        log-intensity standard deviations (sigma_eta_h, sigma_eta_a, cov_eta_ha).
        """
        lam, mu, rho = self.predict_lambdas(home_team_id, away_team_id, crowd=crowd, context_features=context_features)
        _, _, draw_inf = self.compute_context_adjustments(context_features)

        if self.closeness_draw_threshold > 0.0:
            diff = abs(lam - mu)
            if diff < self.closeness_draw_threshold:
                draw_inf = float(self.closeness_draw_boost * (1.0 - (diff / self.closeness_draw_threshold) ** 2))
            else:
                draw_inf = float(self.base_draw_inflation)
                if context_features:
                    draw_inf += float(context_features.get("ref_draw_inf", 0.0))
                    if float(context_features.get("is_derby", 0.0)) > 0.5:
                        draw_inf += 0.04

        if self.param_cov is None:
            return {
                "lam": lam, "mu": mu, "rho": rho,
                "sigma_eta_h": 0.08, "sigma_eta_a": 0.08, "cov_eta_ha": 0.0,
                "draw_inflation": draw_inf,
            }

        n_teams = len(self.team_ids)
        dim = 2 * n_teams + 2
        v_h = np.zeros(dim, dtype=np.float64)
        v_a = np.zeros(dim, dtype=np.float64)

        v_h[2 * n_teams] = 1.0      # mu0
        v_h[2 * n_teams + 1] = 1.0  # home_adv
        v_a[2 * n_teams] = 1.0      # mu0

        h_idx = self.team_to_idx.get(home_team_id)
        a_idx = self.team_to_idx.get(away_team_id)

        extra_var_h = 0.0
        extra_var_a = 0.0

        if h_idx is not None:
            v_h[h_idx] = 1.0
            v_a[n_teams + h_idx] = -1.0
        else:
            extra_var_h += 0.04
            extra_var_a += 0.04

        if a_idx is not None:
            v_h[n_teams + a_idx] = -1.0
            v_a[a_idx] = 1.0
        else:
            extra_var_h += 0.04
            extra_var_a += 0.04

        var_h = max(float(v_h @ self.param_cov @ v_h) + extra_var_h, 1e-6)
        var_a = max(float(v_a @ self.param_cov @ v_a) + extra_var_a, 1e-6)
        cov_ha = float(v_h @ self.param_cov @ v_a)

        return {
            "lam": lam,
            "mu": mu,
            "rho": rho,
            "sigma_eta_h": float(np.sqrt(var_h)),
            "sigma_eta_a": float(np.sqrt(var_a)),
            "cov_eta_ha": cov_ha,
            "draw_inflation": draw_inf,
        }

    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        crowd: str = "full",
        context_features: Optional[dict[str, float]] = None,
        integrate_uncertainty: bool = True,
        max_goals: int = 10,
    ) -> np.ndarray:
        """
        Predict normalized 11x11 joint score matrix.
        By default, integrates over Laplace parameter uncertainty via Gauss-Hermite quadrature.
        """
        unc = self.predict_with_uncertainty(
            home_team_id, away_team_id, crowd=crowd, context_features=context_features
        )
        if integrate_uncertainty:
            return posterior_predictive_score_matrix(
                lam=unc["lam"],
                mu=unc["mu"],
                rho=unc["rho"],
                sigma_eta_h=unc["sigma_eta_h"],
                sigma_eta_a=unc["sigma_eta_a"],
                cov_eta_ha=unc["cov_eta_ha"],
                draw_inflation=unc["draw_inflation"],
                max_goals=max_goals,
            )
        return score_matrix(
            unc["lam"], unc["mu"], unc["rho"],
            draw_inflation=unc["draw_inflation"], max_goals=max_goals
        )
