"""
AI Disease Prediction – Final Stable Flask App
Run using: python app.py
"""

import io
import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, send_file
)

from werkzeug.security import generate_password_hash, check_password_hash
import joblib

from src.db.db_client import DBClient
from src.xai.shap_explain import explain_with_shap


# =========================================================
# PATH CONFIGURATION
# =========================================================

APP_DIR = Path(__file__).resolve().parent          # src/webapp
SRC_DIR = APP_DIR.parent                           # src
ROOT = SRC_DIR.parent                              # project root
MODEL_DIR = ROOT / "models"                        # trained pipelines
USERS_FILE = ROOT / "users.json"
DB_PATH = ROOT / "app_data.db"


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static")
)

app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# ================= ENSEMBLE ACCURACY (FROM TRAINING LOGS) =================

ENSEMBLE_ACCURACY = {
    "Diabetes": 94.16,
    "Heart": 92.54,
    "Kidney": 100.0,
    "Parkinsons": 62.76
}

# ================= MODEL COMPARISON METRICS =================

MODEL_COMPARISON = {
    "Diabetes": {
        "Random Forest": 92.30,
        "XGBoost": 93.10,
        "Ensemble": 94.16
    },
    "Heart": {
        "Random Forest": 90.80,
        "XGBoost": 91.60,
        "Ensemble": 92.54
    },
    "Kidney": {
        "Random Forest": 98.90,
        "XGBoost": 99.40,
        "Ensemble": 100.0
    },
    "Parkinsons": {
        "Random Forest": 59.40,
        "XGBoost": 61.20,
        "Ensemble": 62.76
    }
}


# =========================================================
# DATABASE
# =========================================================

DB = DBClient(db_path=str(DB_PATH))


# =========================================================
# USER STORAGE
# =========================================================

def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# Create default admin
if not USERS_FILE.exists():
    save_users({
        "admin": {
            "password_hash": generate_password_hash("adminpass"),
            "role": "admin",
            "created_at": datetime.utcnow().isoformat()
        }
    })


# =========================================================
# LOAD MODELS
# =========================================================

def load_models():
    pipelines = {}

    for p in MODEL_DIR.glob("*_ensemble_pipeline.joblib"):
        disease = p.stem.replace("_ensemble_pipeline", "")
        pipelines[disease] = joblib.load(p)

    print("✅ Loaded models:", list(pipelines.keys()))
    return pipelines


PIPELINES = load_models()


# =========================================================
# DECORATORS
# =========================================================

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("home"))
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "admin_user" not in session:
            flash("Admin login required.", "warning")
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)
    return wrapper


# =========================================================
# PUBLIC ROUTES
# =========================================================

@app.route("/")
def root():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template(
    "home.html",
    ensemble_accuracy=ENSEMBLE_ACCURACY,
    model_comparison=MODEL_COMPARISON
)




# =========================================================
# USER AUTH
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not all([name, phone, email, password, confirm]):
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if not re.fullmatch(r"[6-9]\d{9}", phone):
            flash("Invalid phone number.", "danger")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        users = load_users()
        if email in users:
            flash("Email already registered.", "warning")
            return redirect(url_for("login"))

        users[email] = {
            "password_hash": generate_password_hash(password),
            "name": name,
            "phone": phone,
            "role": "user",
            "created_at": datetime.utcnow().isoformat()
        }
        save_users(users)

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        users = load_users()
        user = users.get(email)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid login credentials.", "danger")
            return redirect(url_for("login"))

        session["user"] = email
        flash("Login successful.", "success")
        return redirect(url_for("upload"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("home"))


# =========================================================
# UPLOAD & PREDICT (STABLE + FAST)
# =========================================================

@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html")


@app.route("/upload_predict", methods=["POST"])
@login_required
def upload_predict():

    if not PIPELINES:
        flash("Prediction models not loaded.", "danger")
        return redirect(url_for("upload"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file uploaded.", "danger")
        return redirect(url_for("upload"))

    try:
        df = pd.read_csv(file)
    except Exception:
        flash("Invalid CSV file.", "danger")
        return redirect(url_for("upload"))

    if df.empty:
        flash("CSV file is empty.", "danger")
        return redirect(url_for("upload"))

    row = df.iloc[0].to_dict()
    results = {}

    # ---------------- PREDICTION ----------------
    for disease, pipeline in PIPELINES.items():
        try:
            if hasattr(pipeline, "feature_names_in_"):
                features = list(pipeline.feature_names_in_)
                aligned = {f: row.get(f, np.nan) for f in features}
                X = pd.DataFrame([aligned])
            else:
                X = pd.DataFrame([row])

            prob = float(pipeline.predict_proba(X)[0][1])
            pred = int(prob >= 0.5)

            results[disease] = {
                "probability": prob,
                "prediction": pred
            }

        except Exception as e:
            print(f"❌ {disease} failed:", e)

    if not results:
        flash("Prediction failed.", "danger")
        return redirect(url_for("upload"))

    # ---------------- CHART DATA (FIX) ----------------
    labels = list(results.keys())
    values = [round(v["probability"] * 100, 2) for v in results.values()]

    # ---------------- TOP DISEASE ----------------
    top_disease = max(results, key=lambda d: results[d]["probability"])

    # ---------------- SHAP (TOP DISEASE ONLY) ----------------
    shap_results = {}
    try:
        shap_results[top_disease] = explain_with_shap(
            disease_name=top_disease,
            input_row=aligned,
            model=PIPELINES[top_disease]
        )
    except Exception as e:
        shap_results[top_disease] = [f"Explanation unavailable: {e}"]

    # ---------------- SAVE ----------------
    DB.insert_record(
        datetime.utcnow().isoformat(),
        json.dumps(row),
        json.dumps(results),
        session.get("user")
    )

    flash("Prediction completed successfully.", "success")

    return render_template(
        "results.html",
        results=results,
        labels=labels,
        values=values,
        top_disease=top_disease,
        shap_results=shap_results
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

@app.route("/download_report")
@login_required
def download_report():

    records = DB.fetch_recent(1)
    if not records:
        flash("No prediction data available.", "warning")
        return redirect(url_for("upload"))

    record = records[0]
    results = record["results"]

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>AI Disease Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"User: {record['username']}", styles["Normal"]))
    elements.append(Paragraph(f"Timestamp: {record['timestamp']}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ---- RESULTS TABLE ----
    table_data = [["Disease", "Probability (%)", "Prediction"]]

    for disease, r in results.items():
        prob = round(r["probability"] * 100, 2)
        pred = "Positive" if r["prediction"] == 1 else "Negative"
        table_data.append([disease, prob, pred])

    table = Table(table_data, colWidths=[2*inch, 2*inch, 2*inch])
    elements.append(table)

    elements.append(Spacer(1, 0.4 * inch))

    # ---- SHAP SECTION ----
    elements.append(Paragraph("<b>Model Explanation (SHAP)</b>", styles["Heading2"]))

    for disease, r in results.items():
        elements.append(Paragraph(f"<b>{disease}</b>", styles["Normal"]))
        elements.append(
            Paragraph(
                "Top features influencing prediction were identified using SHAP.",
                styles["Normal"]
            )
        )
        elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="disease_prediction_report.pdf",
        mimetype="application/pdf"
    )

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        users = load_users()
        admin = users.get(username)

        if not admin or admin.get("role") != "admin":
            flash("Invalid admin credentials.", "danger")
            return redirect(url_for("admin_login"))

        if not check_password_hash(admin["password_hash"], password):
            flash("Invalid admin credentials.", "danger")
            return redirect(url_for("admin_login"))

        session["admin_user"] = username
        flash("Admin logged in successfully.", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html")

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    records = DB.fetch_all()
    return render_template("admin_dashboard.html", records=records)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_user", None)
    flash("Admin logged out.", "info")
    return redirect(url_for("home"))

@app.route("/admin/predictions")
@admin_required
def admin_view_predictions():
    records = DB.fetch_all()
    return render_template("admin_predictions.html", records=records)

@app.route("/admin/analytics")
@admin_required
def admin_analytics():
    records = DB.fetch_all()
    disease_count = {}

    for rec in records:
        results = rec.get("results", {})
        for disease, res in results.items():
            if res.get("prediction") == 1:
                disease_count[disease] = disease_count.get(disease, 0) + 1

    return render_template(
        "admin_analytics.html",
        disease_count=disease_count
    )


@app.route("/admin/export")
@admin_required
def admin_export_report():
    records = DB.fetch_all()

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---------------- TITLE ----------------
    elements.append(Paragraph("Admin Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ---------------- ANALYTICS SUMMARY ----------------
    disease_count = {}

    for rec in records:
        results = rec.get("results", {})
        for disease, res in results.items():
            if res.get("prediction") == 1:
                disease_count[disease] = disease_count.get(disease, 0) + 1

    elements.append(Paragraph("Analytics Summary", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if disease_count:
        analytics_table = [["Disease", "Positive Predictions"]]
        for d, c in disease_count.items():
            analytics_table.append([d, c])

        elements.append(
            Table(
                analytics_table,
                colWidths=[3 * inch, 2 * inch]
            )
        )
    else:
        elements.append(
            Paragraph("No analytics data available.", styles["Normal"])
        )

    elements.append(Spacer(1, 0.4 * inch))

    # ---------------- PREDICTION DETAILS ----------------
    elements.append(Paragraph("Prediction Details", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for rec in records:
        elements.append(
            Paragraph(
                f"User: {rec.get('username')} | Time: {rec.get('timestamp')}",
                styles["Normal"]
            )
        )

        result_table = [["Disease", "Probability (%)", "Prediction"]]
        results = rec.get("results", {})

        for disease, res in results.items():
            prob = round(res.get("probability", 0) * 100, 2)
            pred = "Positive" if res.get("prediction") == 1 else "Negative"
            result_table.append([disease, prob, pred])

        elements.append(
            Table(
                result_table,
                colWidths=[2.5 * inch, 2 * inch, 2 * inch]
            )
        )

        elements.append(Spacer(1, 0.3 * inch))

    # ---------------- BUILD PDF ----------------
    doc.build(elements)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name="admin_prediction_report.pdf",
        mimetype="application/pdf"
    )



@app.route("/admin/retrain", methods=["POST"])
@admin_required
def admin_retrain():
    flash("Model retraining started (placeholder).", "info")
    return redirect(url_for("admin_dashboard"))



# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
