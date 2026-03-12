"""
Model training pipeline.

Trains separate models for:
  - Moneyline (classification: home win?)
  - Player points (regression)
  - Player rebounds (regression)
  - Player assists (regression)
  - Player PRA (regression)

Uses time-aware splits: train on games before a cutoff, validate on games after.
Saves models to MODEL_DIR.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, timedelta

from sklearn.model_selection import train_test_split

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             log_loss, roc_auc_score, accuracy_score)
import xgboost as xgb

from src.data.ingest import get_connection
from src.features.engineering import (
    player_rolling_features, player_opponent_features,
    player_rest_features, player_starter_signal,
    team_rolling_features, team_defensive_features, team_rest_features,
    STAT_COLS,
)
from src.utils.config import MODEL_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

# Only use last 2 seasons for training — recent data is most relevant
LOOKBACK_YEARS = 2
# Validation split: last N days of data
VAL_DAYS = 60


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def _build_player_training_rows(target_col: str, min_minutes: float = 10.0,
                                 max_rows: int = 80000) -> pd.DataFrame:
    """
    Build a training DataFrame for a player prop model.
    Each row = one player-game with rolling features built from PRIOR games.
    Only uses last ~2 seasons of data for speed and relevance.
    """
    con = get_connection()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_YEARS * 365)
    q = f"""
        SELECT personId, player_name, game_date, gameId,
               points, reboundsTotal, assists, pra,
               threePointersMade, steals, blocks, numMinutes,
               fieldGoalsMade, fieldGoalsAttempted,
               freeThrowsMade, freeThrowsAttempted,
               team_abbrev, opp_abbrev, home
        FROM player_box
        WHERE game_date >= '{cutoff.strftime('%Y-%m-%d')}'
          AND numMinutes >= {min_minutes}
        ORDER BY personId, game_date
    """
    df = con.execute(q).fetchdf()
    if df.empty:
        log.warning("No player data found for training.")
        return pd.DataFrame()

    # Ensure game_date is datetime for comparisons
    df["game_date"] = pd.to_datetime(df["game_date"])

    log.info("Building player training data for '%s' (%d raw rows)...", target_col, len(df))

    rows = []
    player_ids = df["personId"].unique()
    log.info("Processing %d unique players...", len(player_ids))

    for count, pid in enumerate(player_ids):
        if count % 100 == 0 and count > 0:
            log.info("  ... processed %d/%d players (%d rows so far)", count, len(player_ids), len(rows))

        grp = df[df["personId"] == pid].sort_values("game_date").reset_index(drop=True)
        if len(grp) < 15:
            continue

        # Sample every 3rd game to speed up (still captures trends)
        indices = list(range(10, len(grp), 3))
        for i in indices:
            history = grp.iloc[:i].iloc[::-1]  # Most recent first
            target_row = grp.iloc[i]
            target_val = target_row[target_col]
            if pd.isna(target_val):
                continue

            feats = player_rolling_features(history)
            feats["is_home"] = int(target_row.get("home", 0) or 0)

            td = target_row["game_date"]
            target_date = td.date() if hasattr(td, 'date') else pd.Timestamp(td).date()
            feats.update(player_rest_features(history, target_date))
            feats.update(player_starter_signal(history))

            # Opponent history
            opp = target_row.get("opp_abbrev", "")
            vs = history[history["opp_abbrev"] == opp]
            feats.update(player_opponent_features(vs))

            feats["target"] = float(target_val)
            feats["game_date"] = target_row["game_date"]
            rows.append(feats)

            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    result = pd.DataFrame(rows)
    log.info("Built %d training rows for %s", len(result), target_col)
    return result


def _build_game_training_rows(max_rows: int = 15000) -> pd.DataFrame:
    """Build training data for moneyline model. Uses last 2 seasons."""
    con = get_connection()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_YEARS * 365)
    q = f"""
        SELECT gameId, game_date, team_abbrev, opp_abbrev,
               teamScore, opponentScore, home, win,
               assists, reboundsTotal, turnovers,
               fieldGoalsMade, fieldGoalsAttempted,
               threePointersMade, threePointersAttempted,
               freeThrowsMade, freeThrowsAttempted,
               reboundsOffensive, reboundsDefensive
        FROM team_box
        WHERE game_date >= '{cutoff.strftime('%Y-%m-%d')}'
          AND home = 1
        ORDER BY game_date
    """
    df = con.execute(q).fetchdf()
    df["game_date"] = pd.to_datetime(df["game_date"])
    if df.empty:
        return pd.DataFrame()

    log.info("Building game training data (%d home games)...", len(df))

    # Pre-load all team data for the window (+ 90 day buffer for rolling)
    all_team_data = con.execute(f"""
        SELECT * FROM team_box
        WHERE game_date >= '{(cutoff - pd.Timedelta(days=90)).strftime('%Y-%m-%d')}'
        ORDER BY team_abbrev, game_date
    """).fetchdf()
    all_team_data["game_date"] = pd.to_datetime(all_team_data["game_date"])

    rows = []
    team_groups = {name: grp for name, grp in all_team_data.groupby("team_abbrev")}

    for idx, game_row in df.iterrows():
        home_team = game_row["team_abbrev"]
        away_team = game_row["opp_abbrev"]
        gd = game_row["game_date"]

        if not home_team or not away_team:
            continue

        home_hist = team_groups.get(home_team, pd.DataFrame())
        away_hist = team_groups.get(away_team, pd.DataFrame())

        if home_hist.empty or away_hist.empty:
            continue

        home_hist = home_hist[home_hist["game_date"] < gd].sort_values("game_date", ascending=False)
        away_hist = away_hist[away_hist["game_date"] < gd].sort_values("game_date", ascending=False)

        if len(home_hist) < 5 or len(away_hist) < 5:
            continue

        feats = {}
        hf = team_rolling_features(home_hist)
        feats.update({f"home_{k}": v for k, v in hf.items()})
        feats.update({f"home_{k}": v for k, v in team_defensive_features(home_hist).items()})

        target_date = gd.date() if hasattr(gd, 'date') else pd.Timestamp(gd).date()
        feats.update({f"home_{k}": v for k, v in team_rest_features(home_hist, target_date).items()})

        af = team_rolling_features(away_hist)
        feats.update({f"away_{k}": v for k, v in af.items()})
        feats.update({f"away_{k}": v for k, v in team_defensive_features(away_hist).items()})
        feats.update({f"away_{k}": v for k, v in team_rest_features(away_hist, target_date).items()})

        # Differentials
        for col in ["teamScore", "opponentScore"]:
            hk = f"home_team_{col}_avg_10"
            ak = f"away_team_{col}_avg_10"
            if hk in feats and ak in feats:
                feats[f"diff_{col}_10"] = feats[hk] - feats[ak]

        feats["target"] = int(game_row["win"])
        feats["game_date"] = gd
        rows.append(feats)

        if len(rows) >= max_rows:
            break

    result = pd.DataFrame(rows)
    log.info("Built %d game training rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_regression_model(df: pd.DataFrame, name: str) -> dict:
    """Train an XGBoost regressor with time-aware split."""
    df = df.dropna(subset=["target"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    cutoff = df["game_date"].max() - pd.Timedelta(days=VAL_DAYS)

    train = df[df["game_date"] < cutoff].copy()
    val = df[df["game_date"] >= cutoff].copy()

    feat_cols = [c for c in df.columns if c not in ["target", "game_date"]]
    # Keep only numeric columns
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

    X_train = train[feat_cols].fillna(0)
    y_train = train["target"]
    X_val = val[feat_cols].fillna(0)
    y_val = val["target"]

    if len(X_train) < 50:
        log.warning("Not enough training data for %s (%d rows)", name, len(X_train))
        return {}

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              verbose=False)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    log.info("%s — MAE: %.2f, RMSE: %.2f (val=%d rows)", name, mae, rmse, len(val))

    # Save model and feature columns
    model_path = MODEL_DIR / f"{name}.joblib"
    meta = {"feature_cols": feat_cols, "mae": mae, "rmse": rmse,
            "train_size": len(train), "val_size": len(val)}
    joblib.dump({"model": model, "meta": meta}, model_path)
    log.info("Saved %s to %s", name, model_path)

    return meta


def _train_classifier_model(df: pd.DataFrame, name: str) -> dict:
    """Train an XGBoost classifier (calibrated) with time-aware split."""
    df = df.dropna(subset=["target"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    cutoff = df["game_date"].max() - pd.Timedelta(days=VAL_DAYS)

    train = df[df["game_date"] < cutoff].copy()
    val = df[df["game_date"] >= cutoff].copy()

    feat_cols = [c for c in df.columns if c not in ["target", "game_date"]]
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

    X_train = train[feat_cols].fillna(0)
    y_train = train["target"].astype(int)
    X_val = val[feat_cols].fillna(0)
    y_val = val["target"].astype(int)

    if len(X_train) < 50:
        log.warning("Not enough training data for %s (%d rows)", name, len(X_train))
        return {}

    base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Use base model directly (predict_proba from XGBoost is already well-calibrated)
    preds_prob = base.predict_proba(X_val)[:, 1]
    preds_class = (preds_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, preds_class)
    try:
        auc = roc_auc_score(y_val, preds_prob)
    except Exception:
        auc = 0.0
    try:
        ll = log_loss(y_val, preds_prob)
    except Exception:
        ll = 999.0

    log.info("%s — Acc: %.3f, AUC: %.3f, LogLoss: %.3f (val=%d)",
             name, accuracy, auc, ll, len(val))

    model_path = MODEL_DIR / f"{name}.joblib"
    meta = {"feature_cols": feat_cols, "accuracy": accuracy, "auc": auc,
            "log_loss": ll, "train_size": len(train), "val_size": len(val)}
    joblib.dump({"model": base, "meta": meta}, model_path)
    log.info("Saved %s to %s", name, model_path)

    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_all_models() -> dict:
    """Train all models and return metrics."""
    results = {}

    # --- Player prop models ---
    prop_targets = {
        "player_points": "points",
        "player_rebounds": "reboundsTotal",
        "player_assists": "assists",
        "player_pra": "pra",
    }
    for model_name, target_col in prop_targets.items():
        log.info("=== Training %s ===", model_name)
        df = _build_player_training_rows(target_col)
        if not df.empty:
            results[model_name] = _train_regression_model(df, model_name)

    # --- Moneyline model ---
    log.info("=== Training moneyline ===")
    gdf = _build_game_training_rows()
    if not gdf.empty:
        results["moneyline"] = _train_classifier_model(gdf, "moneyline")

    return results


def train_single_model(model_name: str) -> dict:
    """Train a single model by name."""
    prop_targets = {
        "player_points": "points",
        "player_rebounds": "reboundsTotal",
        "player_assists": "assists",
        "player_pra": "pra",
    }

    if model_name in prop_targets:
        df = _build_player_training_rows(prop_targets[model_name])
        if not df.empty:
            return _train_regression_model(df, model_name)
    elif model_name == "moneyline":
        df = _build_game_training_rows()
        if not df.empty:
            return _train_classifier_model(df, model_name)
    else:
        log.error("Unknown model: %s", model_name)
    return {}
