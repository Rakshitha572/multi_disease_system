# src/models/train.py
r"""
Robust training script that trains per-disease models and builds an ensemble
using only the estimators that trained successfully.

Usage:
    python src\models\train.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import sys
import time
import traceback

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

ROOT = Path(".")
PROC_IN = ROOT / "data" / "processed" / "unified_clean_v1.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_TRAIN_ROWS = 20

def load_clean():
    if not PROC_IN.exists():
        raise FileNotFoundError(f"{PROC_IN} not found. Run unify_datasets.py first.")
    return pd.read_csv(PROC_IN)

def get_target_column_for_disease(d):
    mapping = {
        "diabetes": "outcome",
        "heart": "target",
        "kidney": "class",
        "parkinsons": "parkinson_target"
    }
    return mapping.get(d, None)

def safe_metrics(y_true, y_pred, y_prob=None):
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_prob is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    return out

def train_one_disease(disease):
    start_time = time.time()
    print(f"\n=== Training for: {disease} ===")
    df = load_clean()
    if 'disease_label' not in df.columns:
        print("disease_label column missing. Aborting.")
        return
    sub = df[df['disease_label'] == disease].copy()
    if sub.empty:
        print(f"No rows for {disease}. Skipping.")
        return

    target_col = get_target_column_for_disease(disease)
    if target_col is None or target_col not in sub.columns:
        print(f"Target column for '{disease}' not found (expected '{target_col}'). Skipping.")
        return

    preproc_path = MODEL_DIR / f"{disease}_preprocessor.joblib"
    if not preproc_path.exists():
        print(f"Preprocessor not found: {preproc_path}. Run preprocess.py first.")
        return
    prep_obj = joblib.load(preproc_path)
    preprocessor = prep_obj.get("preprocessor")
    feature_cols = prep_obj.get("all_features", [])
    if preprocessor is None or not feature_cols:
        print(f"Invalid preprocessor or feature list for {disease}. Skipping.")
        return

    X = sub[feature_cols]

    # Robust target coercion
    raw_y = sub[target_col]
    y_numeric = pd.to_numeric(raw_y, errors='coerce')
    n_invalid = int(y_numeric.isna().sum())
    if n_invalid > 0:
        print(f"  ⚠ {n_invalid} rows in '{target_col}' could not be converted and will be dropped.")
    valid_mask = ~y_numeric.isna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y_numeric = y_numeric.loc[valid_mask].reset_index(drop=True)

    rounded = np.rint(y_numeric.values).astype(int)
    diff = np.abs(y_numeric.values - rounded)
    if np.any(diff > 1e-6):
        # non-integer target -> label encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_final = pd.Series(le.fit_transform(y_numeric.astype(str)))
        print("  ⚠ Target not integer-like; applied LabelEncoder to create integer labels.")
    else:
        y_final = pd.Series(rounded)

    if len(y_final) < MIN_TRAIN_ROWS:
        print(f"  ⚠ Only {len(y_final)} rows remain after cleaning. Skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=TEST_SIZE, stratify=y_final, random_state=RANDOM_STATE)

    unique, counts = np.unique(y_train, return_counts=True)
    print("  Training class distribution:", dict(zip(unique.tolist(), counts.tolist())))

    # Track which models trained successfully
    trained_estimators = []
    trained_pipelines = {}

    # --- RandomForest ---
    try:
        rf_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        rf_pipeline = Pipeline([('pre', preprocessor), ('clf', rf_clf)])
        rf_pipeline.fit(X_train, y_train)
        trained_estimators.append(('rf', rf_clf))
        trained_pipelines['rf'] = rf_pipeline
        print("  ✓ RandomForest trained")
    except Exception as e:
        print("  ✗ RandomForest training failed:")
        traceback.print_exc()

    # --- XGBoost (optional) ---
    if XGB_AVAILABLE:
        try:
            xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
            xgb_pipeline = Pipeline([('pre', preprocessor), ('clf', xgb_clf)])
            xgb_pipeline.fit(X_train, y_train)
            trained_estimators.append(('xgb', xgb_pipeline.named_steps['clf']))
            trained_pipelines['xgb'] = xgb_pipeline
            print("  ✓ XGBoost trained")
        except Exception as e:
            print("  ✗ XGBoost training failed:")
            traceback.print_exc()
    else:
        print("  ⚠ XGBoost not available; skipping.")

    # --- Logistic Regression ---
    try:
        lr_clf = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
        lr_pipeline = Pipeline([('pre', preprocessor), ('clf', lr_clf)])
        lr_pipeline.fit(X_train, y_train)
        trained_estimators.append(('lr', lr_clf))
        trained_pipelines['lr'] = lr_pipeline
        print("  ✓ LogisticRegression trained")
    except Exception as e:
        print("  ✗ LogisticRegression training failed:")
        traceback.print_exc()

    # If no estimator trained, abort
    if not trained_estimators:
        print("  ✗ No estimators trained successfully — skipping ensemble and saving nothing.")
        return

    # Build ensemble with trained estimators only
    try:
        ensemble = VotingClassifier(estimators=trained_estimators, voting='soft', n_jobs=-1)
        ensemble_pipeline = Pipeline([('pre', preprocessor), ('ensemble', ensemble)])
        ensemble_pipeline.fit(X_train, y_train)
        print("  ✓ Ensemble trained using:", [name for name, _ in trained_estimators])
    except Exception as e:
        print("  ✗ Ensemble training failed:")
        traceback.print_exc()
        ensemble_pipeline = None

    # Evaluate using ensemble if available, else best single estimator
    eval_pipeline = ensemble_pipeline
    if eval_pipeline is None:
        # pick first trained pipeline
        eval_pipeline = next(iter(trained_pipelines.values()))
        print("  ⚠ Evaluating using first trained estimator instead of ensemble.")

    y_pred = eval_pipeline.predict(X_test)
    y_prob = None
    try:
        prob = eval_pipeline.predict_proba(X_test)
        # if binary, take class 1
        if prob.shape[1] == 2:
            y_prob = prob[:, 1]
    except Exception:
        y_prob = None

    metrics = safe_metrics(y_test, y_pred, y_prob)
    print("  Metrics:", metrics)

    # Save successful pipelines & metrics
    for name, pipeline in trained_pipelines.items():
        outpath = MODEL_DIR / f"{disease}_{name}_pipeline.joblib"
        try:
            joblib.dump(pipeline, outpath)
            print(f"  ✓ Saved {name} pipeline -> {outpath}")
        except Exception:
            print(f"  ✗ Failed saving {name} pipeline:")
            traceback.print_exc()

    if ensemble_pipeline is not None:
        try:
            joblib.dump(ensemble_pipeline, MODEL_DIR / f"{disease}_ensemble_pipeline.joblib")
            print(f"  ✓ Saved ensemble pipeline -> {MODEL_DIR / (disease + '_ensemble_pipeline.joblib')}")
        except Exception:
            print("  ✗ Failed saving ensemble pipeline:")
            traceback.print_exc()

    (MODEL_DIR / f"{disease}_metrics.json").write_text(json.dumps(metrics, indent=2))
    elapsed = time.time() - start_time
    print(f"  Finished {disease} in {elapsed:.1f}s")

def main():
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"XGBoost available: {XGB_AVAILABLE}")
    df = load_clean()
    if 'disease_label' not in df.columns:
        raise RuntimeError("unified_clean_v1.csv must contain 'disease_label' column.")
    diseases = df['disease_label'].dropna().unique().tolist()
    print("Found diseases to train:", diseases)
    for d in diseases:
        train_one_disease(d)

if __name__ == "__main__":
    main()
