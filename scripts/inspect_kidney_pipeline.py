# scripts/inspect_kidney_pipeline.py
import joblib, json
p = joblib.load("models/kidney_ensemble_pipeline.joblib")  # or correct filename
print("Pipeline type:", type(p))
# try to find the preprocessor and ensemble inside
pre = getattr(p, "named_steps", {}).get("pre") or getattr(p, "named_steps", {}).get("preprocessor")
ens = getattr(p, "named_steps", {}).get("ensemble") or getattr(p, "named_steps", {}).get("clf")
print("Has preprocessor:", bool(pre))
print("Has ensemble:", bool(ens))
# If pre exists, try to list expected feature names (if stored)
try:
    # If pre is a ColumnTransformer and was fitted, it may have feature_names_in_ or we can try get_feature_names_out
    if hasattr(pre, "get_feature_names_out"):
        print("Transformed feature names (sample):", pre.get_feature_names_out()[:50])
    elif hasattr(pre, "named_transformers_"):
        # show transformer keys and categories for cat encoder
        for k,v in pre.named_transformers_.items():
            print("Transformer:", k, type(v))
            if hasattr(v, "categories_"):
                print(" - categories lengths:", [len(c) for c in v.categories_])
except Exception as e:
    print("Inspect preprocessor failed:", e)

# Print estimators and type
try:
    ests = getattr(ens, "estimators_", None) or getattr(ens, "estimators", None)
    print("Estimators:", ests)
except Exception as e:
    print("Can't read estimators:", e)
