"""Single Source of Truth configuration loader for Süper Lig prediction engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class DixonColesConfig:
    half_life_years: float = 1.25
    season_carryover: float = 0.85
    in_season_boost: float = 1.50
    shot_alpha: float = 0.40
    ridge_att: float = 1.50
    ridge_def: float = 1.50
    squad_prior_weight: float = 0.18
    early_season_prior_weight: float = 0.15
    form_momentum_weight: float = 0.08
    superiority_calibration_scale: float = 1.11
    fixed_rho: Optional[float] = None
    base_draw_inflation: float = 0.0
    closeness_draw_threshold: float = 0.0
    closeness_draw_boost: float = 0.0
    refit_interval: int = 1
    integrate_uncertainty: bool = True
    grid_size: int = 10


@dataclass
class SideMarketsConfig:
    corners_half_life: float = 0.75
    corners_shrinkage_k: float = 6.0
    cards_half_life: float = 0.75
    cards_shrinkage_k: float = 6.0
    in_season_boost: float = 3.0


@dataclass
class MarketBlendConfig:
    w_total: float = 0.25
    w_superiority: float = 0.12
    shin_max_iter: int = 100


@dataclass
class RefereeConfig:
    enabled: bool = True
    shrinkage_k: float = 6.0
    tempo_to_lambda_coef: float = 0.04
    whistle_to_tempo_slope: float = 0.035


@dataclass
class ExternalFactorsConfig:
    weather_enabled: bool = False
    euro_fatigue_enabled: bool = True
    venue_edge_enabled: bool = True
    availability_enabled: bool = True


@dataclass
class AvailabilityConfig:
    max_single_player_lambda_shift: float = 0.05
    max_team_lambda_shift: float = 0.20
    doubtful_p_play: float = 0.50
    prior_weight_minutes: float = 0.85


@dataclass
class BacktestConfig:
    train_seasons: list[str] = field(default_factory=lambda: ["2122", "2223", "2324", "2425"])
    val_season: str = "2526"
    test_season: str = "2627"
    burn_in_matches: int = 300
    refit_interval: int = 10


@dataclass
class ModelConfig:
    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    side_markets: SideMarketsConfig = field(default_factory=SideMarketsConfig)
    market_blend: MarketBlendConfig = field(default_factory=MarketBlendConfig)
    referee: RefereeConfig = field(default_factory=RefereeConfig)
    external_factors: ExternalFactorsConfig = field(default_factory=ExternalFactorsConfig)
    availability: AvailabilityConfig = field(default_factory=AvailabilityConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    raw_dict: dict[str, Any] = field(default_factory=dict)


_GLOBAL_CONFIG: Optional[ModelConfig] = None


def load_model_config(config_path: str = "config/model.yaml") -> ModelConfig:
    """Load model configuration from YAML file with graceful fallback to defaults."""
    global _GLOBAL_CONFIG
    p = Path(config_path)
    if not p.exists():
        # Fallback if run from a subdirectory
        for alt in [Path("../config/model.yaml"), Path("../../config/model.yaml")]:
            if alt.exists():
                p = alt
                break

    data: dict[str, Any] = {}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    dc_cfg = DixonColesConfig(**data.get("dixon_coles", {}))
    sm_cfg = SideMarketsConfig(**data.get("side_markets", {}))
    mb_cfg = MarketBlendConfig(**data.get("market_blend", {}))
    rf_cfg = RefereeConfig(**data.get("referee", {}))
    ef_cfg = ExternalFactorsConfig(**data.get("external_factors", {}))
    av_cfg = AvailabilityConfig(**data.get("availability", {}))
    bt_cfg = BacktestConfig(**data.get("backtest", {}))

    _GLOBAL_CONFIG = ModelConfig(
        dixon_coles=dc_cfg,
        side_markets=sm_cfg,
        market_blend=mb_cfg,
        referee=rf_cfg,
        external_factors=ef_cfg,
        availability=av_cfg,
        backtest=bt_cfg,
        raw_dict=data,
    )
    return _GLOBAL_CONFIG


def get_model_config() -> ModelConfig:
    """Return currently cached model config or load default."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        return load_model_config()
    return _GLOBAL_CONFIG
