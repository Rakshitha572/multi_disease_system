# scripts/inspect_preprocessors.py
"""
Inspect saved preprocessors and ensemble pipelines for diagnostics.
Run: python scripts/inspect_preprocessors.py
"""
import joblib, json
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_DIR = Path("models")

def inspect_preprocessors():
    print("=== Preprocessor metadata ===\n")
    for p in MODEL_DIR.glob("*_preprocessor.joblib"):
        name = p.name.split("_preprocessor.joblib")[0]
        meta = joblib.load(p)
        all_features = meta.get("all_features", [])
        num_cols = meta.get("num_cols", [])
        cat_cols = meta.get("cat_cols", [])
        print(f"Disease: {name}")
        print(f"  all_features (count): {len(all_features)}")
        print(f"  num_cols (count): {len(num_cols)}")
        print(f"  cat_cols (count): {len(cat_cols)}")
        # show first 20 features
        print("  first features:", all_features[:20])
        # check for target-like columns accidentally present
        targets = set(["class","target","outcome","parkinson_target","classification"])
        leaked = [f for f in all_features if f.lower() in targets]
        print("  target-like columns present:", leaked or "None")
        print("")

def inspect_pipelines():
    print("\n=== Ensemble pipelines internals ===\n")
    for p in MODEL_DIR.glob("*_ensemble_pipeline.joblib"):
        name = p.name.split("_ensemble_pipeline.joblib")[0]
        pipeline = joblib.load(p)
        print(f"Disease: {name}")
        print("  pipeline.named_steps keys:", list(pipeline.named_steps.keys()))
        # try to examine the ensemble / clf
        ensemble = pipeline.named_steps.get("ensemble") or pipeline.named_steps.get("clf")
        if ensemble is None:
            print("  No ensemble or clf found in pipeline.named_steps")
        else:
            print("  estimator type:", type(ensemble).__name__)
            # try to list underlying estimators for VotingClassifier
            try:
                estimators = getattr(ensemble, "estimators_", None) or getattr(ensemble, "estimators", None)
                print("  estimators attribute:", type(estimators))
                if estimators:
                    for nm, est in estimators:
                        est_name = type(est).__name__
                        print(f"    - {nm} -> {est_name}")
                else:
                    # sometimes estimators_ populated after fit; try estimators param
                    ests = getattr(ensemble, "estimators", None)
                    if ests:
                        for nm, est in ests:
                            print(f"    - {nm} -> {type(est).__name__}")
            except Exception as e:
                print("  Could not read ensemble.estimators:", e)
        print("")

if __name__ == "__main__":
    inspect_preprocessors()
    inspect_pipelines()
