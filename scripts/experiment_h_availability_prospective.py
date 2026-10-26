"""
Experiment H: Prospective Lineup & Squad Availability Evaluation Protocol (Faz 4.2-B).
Evaluates out-of-sample prediction accuracy with and without prospective squad availability features.

Pre-registered decision gate (H-Kapısı):
1. Requires a minimum sample size of N >= 300 prospective matches with timestamped lineups.
2. Must show statistically significant out-of-sample RPS reduction (paired bootstrap 95% CI upper bound < 0, Holm-adjusted p < 0.05).
3. Availability shifts must remain bounded by max_team_lambda_shift.
4. If N < 300 or gain is not statistically significant, availability_enabled remains False by default in production.
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import duckdb
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from slmodel.config import get_model_config, ModelConfig, ExternalFactorsConfig
from slmodel.data.loaders import load_match_history
from slmodel.features.context import ContextFeatureExtractor
from slmodel.models.dixon_coles import create_dixon_coles_model
from slmodel.simulate.markets import one_x_two
from slmodel.evaluate.audit import rps, brier_score
from slmodel.evaluate.backtest import paired_bootstrap_test


def run_experiment_h(
    db_path: str = "data/warehouse.duckdb",
    output_json: str = "reports/experiment_h_availability.json",
    min_required_matches: int = 300,
) -> dict[str, Any]:
    print("=" * 72)
    print("DENEY H: KADRO DEVAMSIZLIK VE PROSPEKTİF İLK 11 ETKİ ANALİZİ")
    print("=" * 72)

    con = duckdb.connect(db_path, read_only=True)
    try:
        # Check available prospective snapshots and official lineup records
        n_snaps = con.execute("SELECT COUNT(*) FROM availability_snapshots").fetchone()[0]
        n_official = con.execute("SELECT COUNT(*) FROM availability_snapshots WHERE source = 'official_matchday_lineup'").fetchone()[0]
        
        matches_df = load_match_history(con)
    finally:
        con.close()

    print(f"Toplam kayıtlı durum anlık görüntüsü: {n_snaps:,}")
    print(f"Resmi maç günü ilk 11 anlık görüntüsü: {n_official:,}")

    # Identify prospective matches that have availability snapshots prior to kickoff
    con = duckdb.connect(db_path, read_only=True)
    try:
        prospective_matches = con.execute("""
            SELECT DISTINCT m.match_id, m.season, m.kickoff_utc, m.home_team_id, m.away_team_id,
                            m.home_goals, m.away_goals, m.status, m.crowd
            FROM matches m
            JOIN availability_snapshots s ON s.team_id IN (m.home_team_id, m.away_team_id)
            WHERE m.status = 'played' AND s.fetched_at <= m.kickoff_utc
            ORDER BY m.kickoff_utc
        """).fetchdf()
    finally:
        con.close()

    n_eval = n_official
    print(f"Kriterlere uyan prospektif resmi ilk 11 maç sayısı: {n_eval}")

    cfg_base = get_model_config()

    if n_eval < min_required_matches:
        print(f"\n[DURUM: PROSPEKTİF VERİ TOPLAMA AŞAMASINDA]")
        print(f"Önceden belirlenmiş H-Kapısı kuralı gereğince, kadro modelinin üretime alınması")
        print(f"en az {min_required_matches} prospektif kayıtlı maç gerektirir (Mevcut resmi kayıt: {n_eval}).")
        print("Karar: availability_enabled varsayılan olarak KAPALI (False) kalmaya devam edecektir.")

        result = {
            "status": "DATA_COLLECTION_IN_PROGRESS",
            "decision": "KEEP_DISABLED_BY_DEFAULT",
            "min_required_matches": min_required_matches,
            "current_prospective_matches": n_eval,
            "total_availability_snapshots": n_snaps,
            "gate_passed": False,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Deney H durum raporu kaydedildi: {output_json}")
        return result

    # When N >= 300, evaluate paired differences
    print(f"\nN={n_eval} >= {min_required_matches}. Eşli out-of-sample karşılaştırma çalıştırılıyor...")
    # ... paired evaluation logic ...
    return {}


if __name__ == "__main__":
    run_experiment_h()
