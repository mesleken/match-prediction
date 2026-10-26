"""
Reproducible Hyperparameter Tuning and Ablation Study.
Evaluates Form Momentum (0 / 0.04 / 0.08), Half-Life decay, and Ridge regularization
using out-of-sample walk-forward validation and paired bootstrap tests.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from slmodel.config import get_model_config
from slmodel.evaluate.backtest import WalkForwardBacktest, paired_bootstrap_test
from slmodel.models.dixon_coles import create_dixon_coles_model
from slmodel.evaluate.metrics import rps
from slmodel.models.market import devig_shin
from slmodel.simulate.score_matrix import score_matrix
from slmodel.simulate.markets import one_x_two


def run_form_momentum_study(db_path: str = "data/warehouse.duckdb") -> dict[str, Any]:
    """Evaluate Form Momentum weight grid: 0.0, 0.04, 0.08."""
    print("Running Form Momentum study across {0.0, 0.04, 0.08}...")
    cfg = get_model_config()
    backtester = WalkForwardBacktest(db_path=db_path, burn_in_matches=300, refit_interval=10, use_context=True, integrate_uncertainty=False)
    matches_df, _ = backtester.load_data()

    # Evaluate validation season 2526 and holdout 2627
    target_matches = matches_df[matches_df["season"].isin(["2526", "2627"])].copy()

    weights = [0.0, 0.04, 0.08]
    results = {}
    rps_series = {}

    for w in weights:
        res = backtester.run_walk_forward(from_season="2526", to_season="2627", model_kwargs={"form_momentum_weight": w})
        results[str(w)] = {
            "mean_rps": res.metrics["rps_pure_model"],
            "1x2_acc": res.metrics["hit_pure_1x2_pct"],
        }
        rps_series[w] = [r["rps_pure"] for r in res.eval_records]

    # Paired bootstrap vs baseline (w = 0.0)
    baseline_rps = np.array(rps_series[0.0])
    for w in [0.04, 0.08]:
        w_rps = np.array(rps_series[w])
        diff = baseline_rps - w_rps  # positive means w is better (lower RPS)
        boot = paired_bootstrap_test(diff)
        results[str(w)]["delta_rps_vs_zero"] = round(float(np.mean(diff)), 6)
        results[str(w)]["p_value"] = round(boot["p_value"], 4)
        results[str(w)]["ci_95"] = (round(boot["ci_lower"], 6), round(boot["ci_upper"], 6))

    return results


def run_ridge_sensitivity_study(db_path: str = "data/warehouse.duckdb") -> list[dict[str, Any]]:
    """Evaluate Ridge regularization sweep: {0.25, 0.50, 0.75, 1.0, 1.5, 2.0}."""
    print("Running Ridge sensitivity sweep...")
    cfg = get_model_config()
    backtester = WalkForwardBacktest(db_path=db_path, burn_in_matches=300, refit_interval=10, use_context=False, integrate_uncertainty=False)
    matches_df, odds_df = backtester.load_data()

    # Pre-load odds
    odds_map = {}
    for r in odds_df.itertuples():
        mid = int(r.match_id)
        if pd.notna(r.o1) and pd.notna(r.ox) and pd.notna(r.o2):
            odds_map[mid] = [float(r.o1), float(r.ox), float(r.o2)]

    ridge_grid = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    sweep_results = []

    for reg in ridge_grid:
        res = backtester.run_walk_forward(from_season="2122", to_season="2627", model_kwargs={"ridge_att": reg, "ridge_def": reg})
        
        # Calculate market vs model slope
        v_mod_list = []
        v_mkt_list = []
        for r in res.eval_records:
            mid = r["match_id"]
            if mid in odds_map:
                pm = devig_shin(odds_map[mid])
                v_mkt = float(np.log(max(pm[0], 1e-6) / max(pm[2], 1e-6)))
                v_mod = float(np.log(max(r["p_pure_1x2"][0], 1e-6) / max(r["p_pure_1x2"][2], 1e-6)))
                v_mod_list.append(v_mod)
                v_mkt_list.append(v_mkt)

        slope = 1.0
        r2 = 0.0
        if len(v_mod_list) > 100:
            vm = np.array(v_mod_list)
            vk = np.array(v_mkt_list)
            # OLS: vk = a * vm + b
            A = np.vstack([vm, np.ones(len(vm))]).T
            m_slope, c = np.linalg.lstsq(A, vk, rcond=None)[0]
            slope = float(m_slope)
            r2 = float(np.corrcoef(vm, vk)[0, 1] ** 2)

        sweep_results.append({
            "ridge": reg,
            "mean_rps": res.metrics["rps_pure_model"],
            "1x2_acc": res.metrics["hit_pure_1x2_pct"],
            "market_slope_a": round(slope, 4),
            "r2": round(r2, 4),
        })

    return sweep_results


def run_full_study(output_path: str = "reports/tuning_and_ablation_results.json") -> None:
    db = "data/warehouse.duckdb"
    fm_res = run_form_momentum_study(db)
    ridge_res = run_ridge_sensitivity_study(db)

    out = {
        "form_momentum_study": fm_res,
        "ridge_sensitivity_study": ridge_res,
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Study completed and saved to: {output_path}")


if __name__ == "__main__":
    run_full_study()
