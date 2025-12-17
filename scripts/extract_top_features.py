# scripts/extract_top_features.py
"""
Compute and save top-K important features per disease using the RandomForest pipeline.
Saves files models/<disease>_top_features.txt
Run: python scripts/extract_top_features.py
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_DIR = Path("models")
K = 15

def main():
    for rf_path in MODEL_DIR.glob("*_rf_pipeline.joblib"):
        disease = rf_path.name.split("_rf_pipeline.joblib")[0]
        print("Processing", disease)
        pipe = joblib.load(rf_path)
        # pipe: ('pre', preprocessor), ('clf', RandomForestClassifier)
        pre = pipe.named_steps.get('pre')
        clf = pipe.named_steps.get('clf')
        # get original feature list from preprocessor meta file
        meta_path = MODEL_DIR / f"{disease}_preprocessor.joblib"
        meta = joblib.load(meta_path)
        features = meta.get("all_features", [])
        # Transform a small sample (we need number of transformed features)
        try:
            # compute feature importances from RF (on transformed space if necessary)
            importances = clf.feature_importances_
            # If preprocessor one-hot expanded features, we need transformed feature names
            # Try to reconstruct transformed feature names:
            transformed = []
            num_cols = meta.get("num_cols", []) or []
            cat_cols = meta.get("cat_cols", []) or []
            transformed.extend(num_cols)
            if cat_cols:
                enc = pre.named_transformers_['cat'].named_steps['onehot']
                for i, c in enumerate(cat_cols):
                    for cat in enc.categories_[i]:
                        transformed.append(f"{c}__{cat}")
            if not transformed:
                transformed = features
            if len(importances) != len(transformed):
                print("  Warning: importances length != transformed names length ({} != {})".format(len(importances), len(transformed)))
                # fallback: map importances to first N features
                transformed = transformed[:len(importances)]
            # get top K
            idx = np.argsort(importances)[::-1][:K]
            top = [(transformed[i], float(importances[i])) for i in idx]
            outp = MODEL_DIR / f"{disease}_top_{K}_features.txt"
            with open(outp, "w", encoding="utf-8") as f:
                for name, imp in top:
                    f.write(f"{name}\t{imp:.6f}\n")
            print(f"  Saved top features -> {outp}")
        except Exception as e:
            print("  Failed to compute importances for", disease, ":", e)

if __name__ == "__main__":
    main()
