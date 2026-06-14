"""
Model Router for Praso Credit Risk.

Implements scoring-based dispatch: computes Score A (data richness) and
Score B (profile complexity), then routes each client to the tier that
best matches the available information.

Routing tiers:
  BEHAVIORAL    – client has order history; use Behavioral Model
  APPLICATION   – clear profile, no order history; use Application Model
  MANUAL_REVIEW – borderline profile, no order history; flag for human review
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Feature engineering helpers (must match training notebooks 02 and 03)
# ---------------------------------------------------------------------------

def _parse_interval_midpoint(series: pd.Series) -> pd.Series:
    def _mid(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lstrip("([").rstrip("])")
        parts = [p.strip() for p in s.split(",")]
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan
    return series.apply(_mid)


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a raw dataframe.
    Produces all columns expected by both the Application and Behavioral models.

    Input: dataframe matching the schema of 03_behavioral_combined_*.csv
    Output: engineered dataframe (raw columns preserved; derived columns added)
    """
    df = df_raw.copy()

    # Interval columns → numeric midpoints
    for col, new_col in [
        ("capital_social",        "capital_social_mid"),
        ("idade_cnpj",            "idade_cnpj_mid"),
        ("google_maps_avaliacao", "google_maps_avaliacao_mid"),
    ]:
        if col in df.columns:
            df[new_col] = _parse_interval_midpoint(df[col])

    if "cnae_codigo" in df.columns:
        df["cnae_divisao"] = df["cnae_codigo"].astype(str).str.extract(r"^(\d+)")[0]

    if "serasa_credores" in df.columns:
        df["serasa_n_setores"] = df["serasa_credores"].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )

    # Behavioral derived features
    for col in ["valor_mean", "valor_max", "valor_min", "valor_median", "valor_std"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    if "pct_orders_delayed" in df.columns:
        df["has_any_delay"] = (df["pct_orders_delayed"].fillna(0) > 0).astype(int)

    if "delay_max" in df.columns and "delay_mean" in df.columns:
        df["delay_spike_ratio"] = df["delay_max"] / (df["delay_mean"] + 1)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_data_richness(df: pd.DataFrame) -> pd.Series:
    """
    Score A — Data Richness.

    0.0              if client has no order history (orders_count is null)
    [0.5, 1.0]       if client has order history, scaled by orders_per_month
                     (0.5 + 0.5 × min(1, orders_per_month / 10))
    """
    has_data = df["orders_count"].notna()
    score = pd.Series(0.0, index=df.index, name="score_A")
    if has_data.any():
        opm = df.loc[has_data, "orders_per_month"].clip(upper=10)
        score[has_data] = 0.5 + 0.5 * (opm / 10)
    return score


def score_profile_complexity(df: pd.DataFrame, app_model) -> pd.Series:
    """
    Score B — Profile Complexity.

    Runs the Application Model to get p_app, then:
      score_B = 1 − |p_app − 0.5| × 2

    Returns 1.0 for maximally uncertain clients (p_app ≈ 0.5),
    0.0 for very clear profiles (p_app ≈ 0 or 1).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_app = app_model.predict_proba(df)[:, 1]
    score = pd.Series(1.0 - np.abs(p_app - 0.5) * 2, index=df.index, name="score_B")
    return score


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Dispatches clients to the Application Model, Behavioral Model, or
    Manual Review based on Score A (data richness) and Score B (complexity).

    Usage
    -----
    router = ModelRouter.from_latest()
    df_engineered = engineer_features(df_raw)
    result = router.predict(df_engineered)
    # result: DataFrame with columns [tier, score_A, score_B, default_prob, model_used]
    """

    TIERS = ("BEHAVIORAL", "APPLICATION", "MANUAL_REVIEW")

    def __init__(
        self,
        app_model,
        behav_model,
        threshold_a: float = 0.5,
        threshold_b: float = 0.6,
        ml_router=None,
        ml_router_features: Optional[list] = None,
    ):
        self.app_model    = app_model
        self.behav_model  = behav_model
        self.threshold_a  = threshold_a
        self.threshold_b  = threshold_b
        self.ml_router    = ml_router
        self.ml_router_features = ml_router_features

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_latest(cls, threshold_a: float = 0.5, threshold_b: float = 0.6) -> "ModelRouter":
        """Load the most recently saved application and behavioral models."""
        app_files   = sorted(MODELS_DIR.glob("04_application_best_tuned_xgboost_*.joblib"))
        behav_files = sorted(MODELS_DIR.glob("05_behavioral_best_*.joblib"))
        ml_files    = sorted(MODELS_DIR.glob("06_ml_router_decision_tree_*.joblib"))

        if not app_files:
            raise FileNotFoundError("No Application Model found in models/")
        if not behav_files:
            raise FileNotFoundError("No Behavioral Model found in models/")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app_model   = joblib.load(app_files[-1])
            behav_model = joblib.load(behav_files[-1])
            ml_router   = joblib.load(ml_files[-1]) if ml_files else None

        return cls(
            app_model=app_model,
            behav_model=behav_model,
            threshold_a=threshold_a,
            threshold_b=threshold_b,
            ml_router=ml_router,
        )

    # ------------------------------------------------------------------
    # Core scoring and routing
    # ------------------------------------------------------------------

    def score_data_richness(self, df: pd.DataFrame) -> pd.Series:
        return score_data_richness(df)

    def score_profile_complexity(self, df: pd.DataFrame) -> pd.Series:
        return score_profile_complexity(df, self.app_model)

    def route(self, df: pd.DataFrame, use_ml_router: bool = False) -> pd.Series:
        """
        Assign a routing tier to each row.

        Returns a Series with values 'BEHAVIORAL', 'APPLICATION', or 'MANUAL_REVIEW'.
        """
        s_a = self.score_data_richness(df)
        s_b = self.score_profile_complexity(df)

        if use_ml_router and self.ml_router is not None:
            return self._route_ml(df, s_a, s_b)

        return self._route_rules(s_a, s_b)

    def _route_rules(self, s_a: pd.Series, s_b: pd.Series) -> pd.Series:
        tier = pd.Series("APPLICATION", index=s_a.index, name="tier")
        tier[s_a >= self.threshold_a] = "BEHAVIORAL"
        tier[(s_a < self.threshold_a) & (s_b >= self.threshold_b)] = "MANUAL_REVIEW"
        return tier

    def _route_ml(self, df: pd.DataFrame, s_a: pd.Series, s_b: pd.Series) -> pd.Series:
        """ML router: only applies to clients with behavioral data; falls back to rules otherwise."""
        tier = self._route_rules(s_a, s_b)
        has_data = df["orders_count"].notna()
        if not has_data.any():
            return tier

        feats = pd.DataFrame({
            "score_A":            s_a[has_data],
            "score_B":            s_b[has_data],
            "orders_per_month":   df.loc[has_data, "orders_per_month"],
            "pct_orders_delayed": df.loc[has_data, "pct_orders_delayed"],
            "delay_mean":         df.loc[has_data, "delay_mean"],
            "p_app": pd.Series(
                self.app_model.predict_proba(df[has_data])[:, 1],
                index=df[has_data].index,
            ),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml_decisions = self.ml_router.predict(feats)
        tier[has_data] = ml_decisions
        return tier

    # ------------------------------------------------------------------
    # End-to-end predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame, use_ml_router: bool = False) -> pd.DataFrame:
        """
        Score all clients and return a DataFrame with routing and prediction results.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered features (output of engineer_features()).
        use_ml_router : bool
            If True and an ML router is loaded, use it for behavioral clients.

        Returns
        -------
        pd.DataFrame with columns:
            tier, score_A, score_B, default_prob, model_used
        """
        s_a  = self.score_data_richness(df)
        s_b  = self.score_profile_complexity(df)
        tier = self._route_ml(df, s_a, s_b) if (use_ml_router and self.ml_router) \
               else self._route_rules(s_a, s_b)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_app = pd.Series(
                self.app_model.predict_proba(df)[:, 1], index=df.index
            )

        default_prob = p_app.copy()
        model_used   = pd.Series("Application Model", index=df.index)

        behav_mask = tier == "BEHAVIORAL"
        if behav_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_b = self.behav_model.predict_proba(df[behav_mask])[:, 1]
            default_prob[behav_mask] = p_b
            model_used[behav_mask]   = "Behavioral Model"

        model_used[tier == "MANUAL_REVIEW"] = "Application Model (flagged for human review)"

        return pd.DataFrame({
            "tier":         tier,
            "score_A":      s_a.round(4),
            "score_B":      s_b.round(4),
            "default_prob": default_prob.round(4),
            "model_used":   model_used,
        })
