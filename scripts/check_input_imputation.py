# scripts/check_input_imputation.py
"""
Given a JSON-like record, report provided vs imputed features per disease.
Run:
  python scripts/check_input_imputation.py
"""
import joblib, json
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_DIR = Path("models")

# Example input you used:
USER = {
  "age": 50,
  "glucose": 120,
  "bloodpressure": 72,
  "bmi": 30.1,
  "pregnancies": 1,
  "sex": 1,
  "chol": 200,
  "bp": 80,
  "hemo": 12.0
}

def report():
    for p in MODEL_DIR.glob("*_preprocessor.joblib"):
        disease = p.name.split("_preprocessor.joblib")[0]
        meta = joblib.load(p)
        features = meta.get("all_features", [])
        num_cols = meta.get("num_cols", [])
        cat_cols = meta.get("cat_cols", [])
        # normalize keys
        user_norm = {k.lower(): v for k,v in USER.items()}
        provided = []
        imputed = []
        for f in features:
            if f.lower() in user_norm and user_norm[f.lower()] not in (None, "", []):
                provided.append(f)
            else:
                imputed.append(f)
        print(f"\nDisease: {disease}")
        print(f"  Provided count: {len(provided)}")
        print(f"  Required count: {len(features)}")
        print(f"  Provided features: {provided}")
        print(f"  Imputed (first 30): {imputed[:30]}")
        # compute fraction provided
        print(f"  Fraction provided: {len(provided)}/{len(features)} = {len(provided)/max(1,len(features)):.2%}")

if __name__ == "__main__":
    report()
