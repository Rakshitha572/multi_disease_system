# scripts/print_transformed_names.py
import joblib
from pathlib import Path
MODEL_DIR = Path("models")
for f in MODEL_DIR.glob("*_ensemble_pipeline.joblib"):
    name = f.name.split("_ensemble_pipeline.joblib")[0]
    print("===", name, "===")
    pl = joblib.load(f)
    pre = pl.named_steps.get("pre") or pl.named_steps.get("preprocessor")
    if pre is None:
        print("no pre step")
        continue
    try:
        names = pre.get_feature_names_out()
        print("total transformed features:", len(names))
        for i,n in enumerate(names[:120]):
            print(f"{i:03d}", n)
        print("...")
    except Exception as e:
        print("get_feature_names_out failed:", e)
        # try to inspect named_transformers_ keys
        if hasattr(pre, "named_transformers_"):
            for k,t in pre.named_transformers_.items():
                print("transformer:", k, type(t))
