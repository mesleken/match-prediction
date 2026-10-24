"""Statistical evaluation metrics: Ranked Probability Score (RPS), Log-Loss, Brier, ECE, CLV."""

import numpy as np

def rps(p: np.ndarray | list[float], outcome_idx: int) -> float:
    """
    Ranked Probability Score for ordered categorical outcomes (0: Home, 1: Draw, 2: Away).
    Lower is better.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    # Cumulative probability vector
    cum_p = np.cumsum(p_arr)
    # Cumulative observation vector (one-hot indicator)
    obs = np.zeros(len(p_arr), dtype=np.float64)
    obs[outcome_idx] = 1.0
    cum_o = np.cumsum(obs)

    # For K=3 outcomes, sum over K-1 = 2 items divided by (K-1) = 2
    diff = cum_p[:-1] - cum_o[:-1]
    return float(np.sum(diff ** 2) / (len(p_arr) - 1.0))

def score_log_loss(M: np.ndarray, home_goals: int, away_goals: int) -> float:
    """
    Exact scoreline negative log predictive probability: -log M[x, y].
    """
    max_idx = M.shape[0] - 1
    h = min(int(home_goals), max_idx)
    a = min(int(away_goals), max_idx)
    p = max(float(M[h, a]), 1e-12)
    return float(-np.log(p))

def multi_class_log_loss(p: np.ndarray | list[float], outcome_idx: int) -> float:
    """Multi-class cross-entropy log loss: -log p[outcome]."""
    p_arr = np.asarray(p, dtype=np.float64)
    val = max(float(p_arr[outcome_idx]), 1e-12)
    return float(-np.log(val))

def brier_score(p: np.ndarray | list[float], outcome_idx: int) -> float:
    """Multi-class Brier score: sum_k (p_k - y_k)^2."""
    p_arr = np.asarray(p, dtype=np.float64)
    obs = np.zeros(len(p_arr), dtype=np.float64)
    obs[outcome_idx] = 1.0
    return float(np.sum((p_arr - obs) ** 2))

def expected_calibration_error(
    confidences: np.ndarray | list[float],
    correct_indicators: np.ndarray | list[int | bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width probability bins.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct_indicators, dtype=np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(conf)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(corr[in_bin]))
            bin_conf = float(np.mean(conf[in_bin]))
            ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return float(ece)

def closing_line_value(taken_odds: float, closing_odds: float) -> float:
    """
    Closing Line Value (CLV).
    Positive indicates the model achieved better odds than the closing line.
    """
    if closing_odds <= 1.0 or taken_odds <= 1.0:
        return 0.0
    return float((taken_odds / closing_odds) - 1.0)
