# scripts/check_pipeline_shapes.py
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
for f in MODEL_DIR.glob("*_ensemble_pipeline.joblib"):
    name = f.name.split("_ensemble_pipeline.joblib")[0]
    print("===", name, "===")
    pl = joblib.load(f)
    pre = None
    try:
        pre = pl.named_steps.get("pre") or pl.named_steps.get("preprocessor")
    except Exception:
        pre = None
    ens = pl.named_steps.get("ensemble") or pl.named_steps.get("clf") or pl
    print("pipeline type:", type(pl))
    print("has pre:", bool(pre))
    try:
        expected = getattr(ens, "n_features_in_", None)
        print("classifier n_features_in_:", expected)
    except Exception as e:
        print("couldn't read n_features_in_: ", e)
    # if pre exists, build dummy row using pre.feature_names_in_ or saved all_features
    features = None
    if hasattr(pre, "feature_names_in_"):
        features = list(pre.feature_names_in_)
        print("pre.feature_names_in_ count:", len(features))
    else:
        # try call get_feature_names_out (some transformers)
        try:
            names = pre.get_feature_names_out()
            print("pre.get_feature_names_out count:", len(names))
        except Exception as e:
            print("pre.get_feature_names_out failed:", e)
    if features is not None:
        # transform a dummy row
        import pandas as pd, numpy as np
        X = pd.DataFrame([{c: np.nan for c in features}])
        try:
            Xt = pre.transform(X)
            print("pre.transform -> shape:", getattr(Xt, "shape", None))
        except Exception as e:
            print("pre.transform failed:", e)
    print()
