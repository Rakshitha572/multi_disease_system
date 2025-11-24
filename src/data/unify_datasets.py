# src/data/unify_datasets.py
r"""
Unify multiple disease CSVs into:
 - data/processed/unified_raw.csv   (direct concat with standardized column names)
 - data/processed/unified_clean_v1.csv (basic cleaning + imputation)

This script handles the sample column layouts you provided for:
 - diabetes1.csv / diabetes2.csv
 - heart1.csv / heart2.csv
 - kidney1.csv / kidney2.csv
 - parkinson1.csv / parkinson2.csv

Run from project root:
  python src\data\unify_datasets.py
"""
import pandas as pd
from pathlib import Path
import numpy as np

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_RAW = OUT_DIR / "unified_raw.csv"
UNIFIED_CLEAN = OUT_DIR / "unified_clean_v1.csv"

def read_if_exists(path):
    return pd.read_csv(path) if path.exists() else None

def clean_column_names(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def prepare_diabetes(df):
    # common diabetes columns in samples: pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age, outcome
    df = clean_column_names(df)
    # rename variants
    rename_map = {
        "bloodpressure": "bloodpressure",
        "pregnancies": "pregnancies",
        "diabetespedigreefunction": "dpf",
        "diabetespedigreefunction.1": "dpf",
        "outcome": "outcome"
    }
    df = df.rename(columns=rename_map)
    # convert 0 insulin or skinthickness to NaN (common encoding)
    for col in ["insulin", "skinthickness"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df['disease_label'] = 'diabetes'
    return df

def prepare_heart(df):
    # two styles: numeric headers (heart1) or descriptive (heart2)
    df = clean_column_names(df)
    # mapping from heart2 style to heart1 style
    map2 = {
        "age": "age",
        "sex": "sex",
        "chestpaintype": "cp",
        "chestpain": "cp",
        "restingbp": "trestbps",
        "trestbps": "trestbps",
        "cholesterol": "chol",
        "chol": "chol",
        "fastingbs": "fbs",
        "fbs": "fbs",
        "restingecg": "restecg",
        "restecg": "restecg",
        "maxhr": "thalach",
        "thalach": "thalach",
        "exerciseangina": "exang",
        "exang": "exang",
        "oldpeak": "oldpeak",
        "st_slope": "slope",
        "st_slope": "slope",
        "st_slope": "slope",
        "slope": "slope",
        "heartdisease": "target",
        "target": "target"
    }
    for c in df.columns:
        if c in map2:
            df = df.rename(columns={c: map2[c]})
    # sex: map M/F to 1/0 if present
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'M':1,'F':0}).fillna(df['sex'])
        # ensure numeric
        df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    # exang: map Y/N to 1/0
    if 'exang' in df.columns:
        df['exang'] = df['exang'].map({'Y':1,'N':0}).fillna(df['exang'])
        df['exang'] = pd.to_numeric(df['exang'], errors='coerce')
    df['disease_label'] = 'heart'
    return df

def prepare_kidney(df):
    df = clean_column_names(df)
    # kidney1 uses short names: bp, sg, al, su, rbc, bu, sc, sod, pot, hemo, wbcc, rbcc, htn, class
    # kidney2 has many more columns; we'll harmonize a useful subset
    rename_map = {
        "bp": "bp",
        "bp.1": "bp",
        "sg": "sg",
        "sg.1": "sg",
        "al": "al",
        "su": "su",
        "rbc": "rbc",
        "rbcc": "rbcc",
        "bu": "bu",
        "sc": "sc",
        "sod": "sod",
        "pot": "pot",
        "hemo": "hemo",
        "wbcc": "wbcc",
        "wc": "wbcc",
        "pcv": "pcv",
        "htn": "htn",
        "class": "class",
        "classification": "class",
        "dm": "dm",
        "cad": "cad",
        "appet": "appet",
        "pe": "pe",
        "ane": "ane"
    }
    for c in df.columns:
        if c in rename_map:
            df = df.rename(columns={c: rename_map[c]})
    # normalize textual fields
    if 'class' in df.columns:
        # map common class labels: 'ckd'/'notckd', 1/0 etc -> binary: 1 = ckd, 0 = notckd
        df['class'] = df['class'].astype(str).str.lower().map({
            'ckd': 1, 'notckd': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, '1.0':1, '0.0':0
        }).fillna(df['class'])
    # Convert 'rbc' values like 'normal'/'abnormal' to categorical code
    if 'rbc' in df.columns:
        df['rbc'] = df['rbc'].astype(str).str.lower().replace({'normal': 'normal', 'abnormal':'abnormal', '': np.nan})
    df['disease_label'] = 'kidney'
    return df

def prepare_parkinsons(df):
    df = clean_column_names(df)
    # drop name column if present
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    # status is target (1 = Parkinson's, 0 = healthy)
    if 'status' in df.columns:
        df = df.rename(columns={'status': 'parkinson_target'})
    df['disease_label'] = 'parkinsons'
    return df

def unify_all():
    all_dfs = []
    # Diabetes (may have duplicates of same exact files)
    for fname in ["diabetes1.csv", "diabetes2.csv", "diabetes.csv"]:
        path = RAW_DIR / fname
        df = read_if_exists(path)
        if df is not None:
            print(f"Loaded {path}")
            all_dfs.append(prepare_diabetes(df))
    # Heart
    for fname in ["heart1.csv", "heart2.csv", "heart.csv"]:
        path = RAW_DIR / fname
        df = read_if_exists(path)
        if df is not None:
            print(f"Loaded {path}")
            all_dfs.append(prepare_heart(df))
    # Kidney
    for fname in ["kidney1.csv", "kidney2.csv", "kidney.csv"]:
        path = RAW_DIR / fname
        df = read_if_exists(path)
        if df is not None:
            print(f"Loaded {path}")
            all_dfs.append(prepare_kidney(df))
    # Parkinsons
    for fname in ["parkinson1.csv", "parkinson2.csv", "parkinsons.csv", "parkinsons1.csv"]:
        path = RAW_DIR / fname
        df = read_if_exists(path)
        if df is not None:
            print(f"Loaded {path}")
            all_dfs.append(prepare_parkinsons(df))
    if not all_dfs:
        print("No input files found in data/raw/. Place your CSVs there and re-run.")
        return
    # Concatenate allowing for different columns (outer join style)
    unified = pd.concat(all_dfs, ignore_index=True, sort=False)
    # lowercase column names already applied in helpers
    print("Unified shape (before clean):", unified.shape)
    # Save raw unified
    unified.to_csv(UNIFIED_RAW, index=False)
    print(f"Saved unified raw to: {UNIFIED_RAW}")

    # ---- Basic cleaning -> unified_clean_v1.csv ----
    clean = unified.copy()

    # Numeric columns: try converting to numeric where sensible
    for col in clean.columns:
        if col in ['disease_label', 'class', 'parkinson_target', 'outcome', 'cp', 'restecg', 'rbc', 'appet']:
            continue
        # try numeric conversion
        clean[col] = pd.to_numeric(clean[col], errors='ignore')

    # Impute numeric NaNs with median per-column
    numeric_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median = clean[col].median()
        if np.isnan(median):
            continue
        clean[col] = clean[col].fillna(median)

    # Impute object/categorical with mode
    obj_cols = clean.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        mode = clean[col].mode(dropna=True)
        if not mode.empty:
            clean[col] = clean[col].fillna(mode[0])

    # Standardize some target columns to numeric binary
    if 'outcome' in clean.columns:
        clean['outcome'] = clean['outcome'].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)
    if 'parkinson_target' in clean.columns:
        clean['parkinson_target'] = clean['parkinson_target'].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)
    if 'class' in clean.columns:
        # ensure class is numeric if convertible
        clean['class'] = pd.to_numeric(clean['class'], errors='ignore')
    if 'target' in clean.columns:
        clean['target'] = pd.to_numeric(clean['target'], errors='coerce').fillna(0).astype(int)

    # final save
    clean.to_csv(UNIFIED_CLEAN, index=False)
    print(f"Saved cleaned unified dataset to: {UNIFIED_CLEAN}")
    # Print basic summary
    print("\n--- Summary ---")
    print("Total rows:", len(clean))
    print("Disease distribution:")
    print(clean['disease_label'].value_counts(dropna=False))

if __name__ == "__main__":
    unify_all()
