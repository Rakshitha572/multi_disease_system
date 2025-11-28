# src/data/preprocess.py
r"""
Per-disease preprocessing pipelines.
Saves a preprocessor (scikit-learn Pipeline) for each disease into models/
Run:
    python src/data/preprocess.py
"""

import pandas as pd
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ROOT = Path(".")
PROC_IN = ROOT / "data" / "processed" / "unified_clean_v1.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_clean():
    if not PROC_IN.exists():
        raise FileNotFoundError(f"{PROC_IN} not found. Run unify_datasets.py first.")
    return pd.read_csv(PROC_IN)

def build_and_save_pipelines():
    df = load_clean()
    diseases = df['disease_label'].dropna().unique().tolist()
    print("Detected diseases:", diseases)

    for d in diseases:
        print(f"\nPreprocessing for: {d}")
        sub = df[df['disease_label'] == d].copy()

        target_names = ['outcome', 'target', 'class', 'parkinson_target']
        features = [c for c in sub.columns if c not in target_names + ['disease_label']]
        features = [f for f in features if f.lower() not in ('name', 'id')]

        num_cols = sub[features].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in features if c not in num_cols]

        print(f"  numeric cols: {len(num_cols)}; categorical cols: {len(cat_cols)}")

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        if cat_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
        else:
            cat_pipeline = None

        transformers = []
        transformers.append(("num", num_pipeline, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipeline, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        X = sub[features]
        preprocessor.fit(X)

        joblib.dump({
            "preprocessor": preprocessor,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "all_features": features
        }, MODEL_DIR / f"{d}_preprocessor.joblib")

        transformed_cols = num_cols.copy()
        if cat_cols:
            enc = preprocessor.named_transformers_['cat'].named_steps['onehot']
            for i, col in enumerate(cat_cols):
                for cat in enc.categories_[i]:
                    transformed_cols.append(f"{col}__{cat}")

        pd.Series(transformed_cols).to_csv(MODEL_DIR / f"{d}_features.csv", index=False, header=False)

        print(f"  ✓ Saved: {d}_preprocessor.joblib")
        print(f"  ✓ Saved: {d}_features.csv")

if __name__ == "__main__":
    build_and_save_pipelines()
