# scripts/generate_shap_summary_images.py
"""
Generate SHAP summary images per disease and save to models/<disease>_shap_summary.png.

Strategy:
 - Load full *_ensemble_pipeline.joblib pipeline.
 - Attempt to get fitted preprocessor to transform a reasonable example row.
 - Prefer to compute SHAP using a tree-based estimator (RandomForest) if present.
 - Fallback: attempt shap.Explainer on whole ensemble, but skip on failure (report error).
"""
import joblib, json, os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import shap
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependency:", e)
    print("Install shap and matplotlib: pip install shap matplotlib")
    sys.exit(1)

ROOT = Path(".")
MODEL_DIR = ROOT / "models"

# small example input (will be used to build DataFrame when we can)
EXAMPLE = {
    "age": 50, "glucose": 120, "bloodpressure": 72, "bmi": 30.1,
    "pregnancies": 1, "sex": 1, "chol": 200, "bp": 80, "hemo": 12.0
}

def safe_save_fig(fig, path):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=150)
    plt.close(fig)

def get_preprocessor_and_feature_names(pipeline):
    # pipeline may be a sklearn Pipeline with named_steps
    pre = None
    feature_names = None
    try:
        pre = pipeline.named_steps.get("pre") or pipeline.named_steps.get("preprocessor")
    except Exception:
        pre = None
    if pre is not None and hasattr(pre, "feature_names_in_"):
        feature_names = list(pre.feature_names_in_)
    # Some projects store metadata in separate preprocessor joblib files; skip here
    return pre, feature_names

def pick_tree_estimator(ensemble):
    """
    If ensemble is a VotingClassifier, try to return a tree-based estimator:
    - Prefer RandomForestClassifier
    - Else return any estimator whose class name contains 'Forest' or 'Tree' or 'XGB'
    """
    # if it's a pipeline containing an 'ensemble' step, pull that
    estimators = None
    try:
        # VotingClassifier stores fitted attribute 'estimators_' or 'named_estimators_'
        estimators_map = getattr(ensemble, "named_estimators_", None) or getattr(ensemble, "estimators_", None)
        if isinstance(estimators_map, dict):
            # named_estimators_
            estimators = list(estimators_map.values())
        elif isinstance(estimators_map, (list, tuple)):
            estimators = list(estimators_map)
        else:
            # maybe ensemble.estimators exists as list of (name, est)
            try:
                raw = getattr(ensemble, "estimators", None) or getattr(ensemble, "estimators_", None)
                if isinstance(raw, list):
                    # items could be (name, est) or est
                    tmp = []
                    for el in raw:
                        if isinstance(el, tuple) and len(el) >= 2:
                            tmp.append(el[1])
                        else:
                            tmp.append(el)
                    estimators = tmp
            except Exception:
                estimators = None
    except Exception:
        estimators = None

    if not estimators:
        # If it's not an ensemble, maybe it's already a single estimator
        return ensemble

    # Prefer RF
    for est in estimators:
        try:
            name = est.__class__.__name__.lower()
            if "randomforest" in name:
                return est
        except Exception:
            pass
    # next prefer other tree-ish estimators
    for est in estimators:
        try:
            name = est.__class__.__name__.lower()
            if any(k in name for k in ("forest", "tree", "xgb", "lgbm", "catboost")):
                return est
        except Exception:
            pass
    # fallback to first estimator
    return estimators[0] if estimators else ensemble

def make_input_df(feature_names):
    if feature_names:
        # try to match feature names with EXAMPLE keys (case-insensitive)
        lower_map = {k.lower(): v for k,v in EXAMPLE.items()}
        row = {fn: lower_map.get(fn.lower(), np.nan) for fn in feature_names}
        return pd.DataFrame([row])
    else:
        return pd.DataFrame([EXAMPLE])

def try_shap_tree(estimator, X_trans, out_path, disease):
    """Attempt to run shap.TreeExplainer on estimator with X_trans (numpy or DataFrame)."""
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans if hasattr(X_trans, "values") else X_trans)
        # shap_values shape: for binary classification could be list [class0, class1] or array
        # We will try to handle both by picking the last column/class
        try:
            if isinstance(shap_values, list):
                sv = shap_values[-1]
            else:
                sv = shap_values
        except Exception:
            sv = shap_values

        fig = shap.summary_plot(sv, X_trans, show=False)
        if fig is None:
            # shap.summary_plot sometimes returns None (it draws directly). Grab current fig.
            fig = plt.gcf()
        safe_save_fig(fig, out_path)
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        return False, tb

def try_shap_explainer_callable(predict_fn, X_trans, out_path):
    """Try shap.Explainer on a callable (predict_fn)."""
    try:
        explainer = shap.Explainer(predict_fn, masker=shap.maskers.Independent(X_trans))
        shap_vals = explainer(X_trans)
        fig = shap.summary_plot(shap_vals.values, X_trans, show=False)
        if fig is None:
            fig = plt.gcf()
        safe_save_fig(fig, out_path)
        return True, None
    except Exception as e:
        return False, traceback.format_exc()

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for p in MODEL_DIR.glob("*_ensemble_pipeline.joblib"):
        disease = p.name.split("_ensemble_pipeline.joblib")[0]
        print(f"\n=== {disease} ===")
        try:
            pl = joblib.load(p)
        except Exception as e:
            print("Failed to load pipeline:", e)
            continue

        pre, feature_names = get_preprocessor_and_feature_names(pl)
        X_df = make_input_df(feature_names)

        # If we have a fitted preprocessor, transform to the numeric matrix the estimator expects
        if pre is not None:
            try:
                X_trans = pre.transform(X_df)
                # If transform returns a 1D vector, wrap to 2D
                if hasattr(X_trans, "shape") and len(X_trans.shape) == 1:
                    X_trans = X_trans.reshape(1, -1)
                # for shap plotting we prefer a DataFrame with column names if possible
                try:
                    colnames = getattr(pre, "get_feature_names_out", None)
                    if callable(colnames):
                        cols = pre.get_feature_names_out()
                        X_for_plot = pd.DataFrame(X_trans, columns=cols)
                    else:
                        # fallback: try to get transformed names from pipeline metadata if present
                        X_for_plot = pd.DataFrame(X_trans)
                except Exception:
                    X_for_plot = pd.DataFrame(X_trans)
            except Exception as e:
                print("pre.transform failed (attempting to coerce and retry):", e)
                # coerce and retry build simple numeric DF
                for c in X_df.columns:
                    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
                try:
                    X_trans = pre.transform(X_df)
                    X_for_plot = pd.DataFrame(X_trans)
                except Exception as e2:
                    print("pre.transform retry failed:", e2)
                    X_trans = X_df.values
                    X_for_plot = pd.DataFrame(X_trans, columns=X_df.columns)
        else:
            print("No preprocessor found; using raw example keys (best-effort).")
            X_trans = X_df.values
            X_for_plot = X_df

        # pick a candidate estimator
        try:
            ensemble = pl.named_steps.get("ensemble") or pl.named_steps.get("clf") or pl
        except Exception:
            ensemble = pl

        est = pick_tree_estimator(ensemble)
        out_path = MODEL_DIR / f"{disease}_shap_summary.png"
        print("Using estimator:", type(est).__name__, "-> saving:", out_path)
        # Try TreeExplainer on chosen estimator
        ok, err = try_shap_tree(est, X_for_plot, out_path, disease)
        if ok:
            print("Saved shap summary using", type(est).__name__)
            continue
        else:
            print("TreeExplainer failed for", type(est).__name__)
            print(err)

        # Fallback: if ensemble has a predict_proba callable, try shap.Explainer on that
        if hasattr(ensemble, "predict_proba"):
            print("Trying shap.Explainer on ensemble.predict_proba (wrapped).")
            def wrapped_proba(x):
                # accept numpy array or DataFrame
                try:
                    xx = x
                    if not isinstance(xx, (pd.DataFrame, np.ndarray)):
                        xx = pd.DataFrame(xx)
                    # ensure correct shape
                    return ensemble.predict_proba(xx)
                except Exception as e:
                    # some classifiers output string arrays - we can't salvage here
                    raise

            ok2, err2 = try_shap_explainer_callable(wrapped_proba, X_for_plot, out_path)
            if ok2:
                print("Saved shap summary using shap.Explainer(ensemble.predict_proba).")
                continue
            else:
                print("shap.Explainer on predict_proba failed:")
                print(err2)

        print("Could not generate SHAP for", disease, "- see errors above. Skipping.")

if __name__ == "__main__":
    main()
