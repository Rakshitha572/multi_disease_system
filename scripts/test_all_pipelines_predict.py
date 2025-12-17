# scripts/test_all_pipelines_predict.py
import joblib, json
import numpy as np, pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")
example = {
    "age": 50, "glucose": 120, "bloodpressure": 72, "bmi": 30.1,
    "pregnancies": 1, "sex": 1, "chol": 200, "bp": 80, "hemo": 12.0
}

for p in MODEL_DIR.glob("*_ensemble_pipeline.joblib"):
    name = p.name.split("_ensemble_pipeline.joblib")[0]
    print("===", name, "===")
    pl = joblib.load(p)
    # try to find original raw feature column names (pre.feature_names_in_)
    try:
        pre = pl.named_steps.get("pre") or pl.named_steps.get("preprocessor")
    except Exception:
        pre = None

    if pre is not None and hasattr(pre, "feature_names_in_"):
        cols = list(pre.feature_names_in_)
        row = {c: example.get(c.lower(), np.nan) for c in cols}
        X = pd.DataFrame([row])
        print("Using pre.feature_names_in_ (count):", len(cols))
    else:
        # fallback to using example keys
        X = pd.DataFrame([example])
        print("Using example keys")

    # coerce pd.NA -> np.nan
    X = X.replace({pd.NA: np.nan})

    try:
        # prefer predict_proba
        prob = None
        try:
            pa = pl.predict_proba(X)
            if pa is not None:
                if pa.shape[1] == 2:
                    prob = float(pa[0,1])
                else:
                    prob = float(pa[0].max())
        except Exception as e:
            print("predict_proba failed:", e)

        if prob is None:
            try:
                pv = pl.predict(X)
                print("predict (fallback) ->", int(pv[0]))
            except Exception as e:
                print("predict fallback failed:", e)
        else:
            print("probability ->", prob)

    except Exception as outer:
        print("ERROR running pipeline:", outer)

    print()
