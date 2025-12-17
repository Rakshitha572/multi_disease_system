"""
Explainable AI using Permutation Feature Importance
Works for ensemble & pipeline models
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def explain_with_shap(
    disease_name,
    input_row: dict,
    model,
    top_k: int = 5
):
    """
    Returns top contributing features using permutation importance
    """

    # Convert input to DataFrame
    X = pd.DataFrame([input_row])

    try:
        result = permutation_importance(
            model,
            X,
            model.predict(X),
            n_repeats=1,
            random_state=42,
            scoring="accuracy"
        )
    except Exception as e:
        return [f"Explanation unavailable: {e}"]

    importances = result.importances_mean
    features = X.columns.tolist()

    ranked = sorted(
        zip(features, importances),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    explanations = []
    for feat, score in ranked:
        explanations.append(
            f"{feat} influenced the {disease_name} prediction "
            f"(importance={round(float(score), 4)}) "
            f"with input value {input_row.get(feat)}"
        )

    return explanations
