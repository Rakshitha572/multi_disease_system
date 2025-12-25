import shap
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline


def _extract_model_and_preprocessor(pipeline: Pipeline):
    """
    Robustly extract preprocessor and model from ANY sklearn pipeline
    """
    steps = pipeline.steps

    model = None
    preprocessor = None

    for name, step in steps:
        cls = step.__class__.__name__.lower()

        # identify model
        if any(k in cls for k in ["xgb", "forest", "logistic", "classifier"]):
            model = step

        # identify preprocessor
        if any(k in cls for k in ["columntransformer", "preprocess", "scaler"]):
            preprocessor = step

    return model, preprocessor


def explain_with_shap(disease_name, input_row, model_dir):
    """
    SHAP explanation for TOP BASE MODEL (XGB > RF)
    """

    model_dir = Path(model_dir)

    # ---------------- LOAD BASE MODEL ----------------
    xgb_path = model_dir / f"{disease_name}_xgb_pipeline.joblib"
    rf_path = model_dir / f"{disease_name}_rf_pipeline.joblib"

    if xgb_path.exists():
        pipeline = joblib.load(xgb_path)
        model_type = "XGBoost"
    elif rf_path.exists():
        pipeline = joblib.load(rf_path)
        model_type = "RandomForest"
    else:
        return ["Explainable AI not available for this disease"]

    # ---------------- EXTRACT COMPONENTS ----------------
    model, preprocessor = _extract_model_and_preprocessor(pipeline)

    if model is None:
        return ["Explainable AI failed: ML model not detected"]

    # ---------------- PREPARE INPUT ----------------
    X = pd.DataFrame([input_row])

    if preprocessor:
        X_transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()
    else:
        X_transformed = X.values
        feature_names = X.columns

    # ---------------- SHAP EXPLAINER ----------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        # binary classifier handling
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        contributions = list(zip(feature_names, shap_vals))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        explanation = [
            f"{f} {'increases' if v > 0 else 'decreases'} risk"
            for f, v in contributions[:5]
        ]

        explanation.insert(0, f"Explained using {model_type} model")

        return explanation

    except Exception as e:
        return [f"SHAP explanation failed: {str(e)}"]
