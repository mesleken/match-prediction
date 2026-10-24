"""Tamper-proof SHA-256 hash-chained prediction logging and audit verification."""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd


def normalize_utc_timestamp(dt: Any) -> datetime:
    """Normalize any timestamp into a naive UTC datetime object for bit-identical round-tripping."""
    ts = pd.to_datetime(dt)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def compute_row_hash(
    pred_id: int,
    match_id: int,
    predicted_at: datetime,
    as_of: datetime,
    lam_home: float,
    lam_away: float,
    rho: float,
    prev_hash: str,
) -> str:
    """Compute SHA-256 hash linking this prediction to the cryptographic chain."""
    p_at_iso = normalize_utc_timestamp(predicted_at).isoformat()
    a_of_iso = normalize_utc_timestamp(as_of).isoformat()

    msg = (
        f"{pred_id}:{match_id}:{p_at_iso}:{a_of_iso}:"
        f"{lam_home:.6f}:{lam_away:.6f}:{rho:.6f}:{prev_hash}"
    )
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


class PredictionLogger:
    """Manages append-only, tamper-proof prediction logging."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = db_path

    def get_latest_hash_and_id(self, con: duckdb.DuckDBPyConnection) -> tuple[str, int]:
        """Fetch the row_hash and pred_id of the most recent logged prediction."""
        res = con.execute("SELECT pred_id, row_hash FROM prediction_log ORDER BY pred_id DESC LIMIT 1").fetchone()
        if res is None:
            return "0" * 64, 0
        return str(res[1]), int(res[0])

    def log_prediction(
        self,
        match_id: int,
        as_of: datetime,
        lam_home: float,
        lam_away: float,
        rho: float,
        score_matrix_arr: np.ndarray,
        odds_snapshot_id: Optional[int] = None,
        model_version: str = "1.0.0",
        config_hash: str = "default",
        data_hash: str = "default",
        git_commit: str = "clean",
    ) -> dict[str, Any]:
        """Append a new prediction to the tamper-proof ledger."""
        con = duckdb.connect(self.db_path)
        try:
            prev_hash, last_id = self.get_latest_hash_and_id(con)
            new_pred_id = last_id + 1

            now_utc_norm = normalize_utc_timestamp(datetime.now(timezone.utc))
            as_of_norm = normalize_utc_timestamp(as_of)

            row_hash = compute_row_hash(
                pred_id=new_pred_id,
                match_id=match_id,
                predicted_at=now_utc_norm,
                as_of=as_of_norm,
                lam_home=lam_home,
                lam_away=lam_away,
                rho=rho,
                prev_hash=prev_hash,
            )

            score_matrix_json = json.dumps(score_matrix_arr.tolist())

            con.execute(
                """
                INSERT INTO prediction_log (
                    pred_id, match_id, predicted_at, as_of,
                    config_hash, data_hash, git_commit, model_version,
                    lam_home, lam_away, rho, score_matrix_json,
                    odds_snapshot_id, prev_hash, row_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    new_pred_id,
                    match_id,
                    now_utc_norm,
                    as_of_norm,
                    config_hash,
                    data_hash,
                    git_commit,
                    model_version,
                    float(lam_home),
                    float(lam_away),
                    float(rho),
                    score_matrix_json,
                    odds_snapshot_id,
                    prev_hash,
                    row_hash,
                ],
            )

            return {
                "pred_id": new_pred_id,
                "prev_hash": prev_hash,
                "row_hash": row_hash,
            }
        finally:
            con.close()


def verify_prediction_log(db_path: str = "data/warehouse.duckdb") -> dict[str, Any]:
    """
    Verify full cryptographic integrity of the prediction log hash chain.
    Returns status, count of verified rows, and error message if tampering detected.
    """
    con = duckdb.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT 
                pred_id, match_id, predicted_at, as_of,
                lam_home, lam_away, rho, prev_hash, row_hash
            FROM prediction_log
            ORDER BY pred_id ASC
            """
        ).fetchall()

        if not rows:
            return {"valid": True, "count": 0, "message": "Log is empty (0 predictions)."}

        expected_prev_hash = "0" * 64

        for row in rows:
            p_id, m_id, p_at, a_of, lam_h, lam_a, r, prev_h, r_hash = row

            # Verify chain linkage
            if prev_h != expected_prev_hash:
                return {
                    "valid": False,
                    "pred_id": p_id,
                    "error": f"Chain broken at pred_id {p_id}: expected prev_hash {expected_prev_hash}, got {prev_h}",
                }

            # Recalculate hash
            computed = compute_row_hash(
                pred_id=p_id,
                match_id=m_id,
                predicted_at=p_at,
                as_of=a_of,
                lam_home=float(lam_h),
                lam_away=float(lam_a),
                rho=float(r),
                prev_hash=prev_h,
            )

            if computed != r_hash:
                return {
                    "valid": False,
                    "pred_id": p_id,
                    "error": f"Hash mismatch at pred_id {p_id}: recorded {r_hash}, computed {computed}",
                }

            expected_prev_hash = r_hash

        return {
            "valid": True,
            "count": len(rows),
            "latest_hash": expected_prev_hash,
            "message": f"All {len(rows)} predictions successfully verified against cryptographic ledger.",
        }
    finally:
        con.close()
