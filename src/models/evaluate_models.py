import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--disease", required=True)
args = parser.parse_args()

disease = args.disease

# Load test data
df = pd.read_csv(f"data/processed/{disease}_test.csv")

X_test = df.drop("target", axis=1)
y_test = df["target"]

models = {
    "Logistic Regression": f"models/{disease}_lr.joblib",
    "Random Forest": f"models/{disease}_rf.joblib",
    "SVM": f"models/{disease}_svm.joblib",
    "Voting Classifier": f"models/{disease}_voting.joblib"
}

print(f"\n🔍 Accuracy Report for {disease.upper()} Dataset\n")

for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:20s} : {acc*100:.2f}%")
