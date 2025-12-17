🧠 Ensemble Based Multi Disease Prediction
Ensemble Learning with Explainable AI (SHAP)
📌 Project Overview

The Ensemble Based Multi Disease Prediction System is a web-based healthcare analytics platform that predicts multiple diseases using ensemble machine learning models and provides explainable predictions using SHAP (SHapley Additive exPlanations).

The system allows users to:

Register and log in securely

Upload medical CSV data

Predict diseases such as Diabetes, Heart Disease, Kidney Disease, and Parkinson’s

View prediction probabilities and visual analytics

Understand model decisions through SHAP explanations

Download prediction reports in PDF format

An Admin dashboard is also provided for monitoring users, predictions, and analytics.

🎯 Objectives

Predict multiple diseases using ensemble learning techniques

Improve prediction accuracy through Voting Classifier

Provide transparency using Explainable AI (SHAP)

Offer an easy-to-use web interface for non-technical users

Maintain secure user authentication and role-based access

Store prediction history for auditing and analysis

🏗️ System Architecture

Frontend

HTML5, CSS3, Bootstrap

Chart.js for data visualization

Backend

Flask (Python)

SQLite database

Machine Learning

Random Forest

XGBoost

Logistic Regression

Ensemble Voting Classifier

Explainable AI

SHAP (feature attribution)

🧪 Diseases Covered

Disease	Model Type
Diabetes	Ensemble (RF + XGB + LR)
Heart Disease	Ensemble (RF + XGB + LR)
Kidney Disease	Ensemble (RF + XGB + LR)
Parkinson’s	Ensemble (RF + XGB + LR)

📊 Model Performance (Accuracy)

Disease	Accuracy
Diabetes	94.15%
Heart Disease	92.54%
Kidney Disease	100%
Parkinson’s	62.76%

Accuracy is calculated after training ensemble models using cross-validation.

👤 User Roles

🔹 User

Register & login

Upload CSV file

View predictions & probabilities

Analyze SHAP explanations

Download PDF reports

🔹 Admin

Secure admin login

View all prediction records

View analytics dashboards

Export admin reports (PDF)

Monitor system usage

📁 Project Folder Structure

multi-disease-prediction/
│
├── models/                         # Trained ML models (.joblib)
│   ├── diabetes_ensemble_pipeline.joblib
│   ├── heart_ensemble_pipeline.joblib
│   ├── kidney_ensemble_pipeline.joblib
│   └── parkinsons_ensemble_pipeline.joblib
│
├── src/
│   ├── models/
│   │   ├── train.py               # Model training script
│   │   └── predict.py             # Standalone prediction logic
│   │
│   ├── db/
│   │   └── db_client.py            # SQLite database handler
│   │
│   ├── xai/
│   │   └── shap_explain.py         # SHAP explanation logic
│   │
│   └── webapp/
│       ├── app.py                  # Main Flask application
│       ├── templates/              # HTML templates
│       └── static/                 # CSS, JS, assets
│
├── users.json                      # User credentials storage
├── app_data.db                    # SQLite database
├── run_dashboard.bat              # Desktop launcher
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies

🚀 How to Run the Project
🔹 Option 1: Run via Desktop (Recommended)

Double-click:

run_dashboard.bat

Then open browser:

http://127.0.0.1:5000

🔹 Option 2: Run via Terminal

cd C:\Projects\multi-disease-prediction
python src/webapp/app.py
🔐 Default Admin Credentials
makefile
Copy code
Username: admin
Password: adminpass

📈 Visual Analytics

Bar chart: Disease probabilities

Line chart: Comparative probabilities

Histogram: Probability distribution

SHAP feature importance per disease

🧾 Report Generation

Users can download prediction reports (PDF)

Admin can export analytics and prediction summaries

Reports include:

Disease predictions

Probabilities

Charts

SHAP explanations

🔍 Explainable AI (SHAP)

SHAP explanations provide:

Feature contribution to prediction

Positive & negative influence analysis

Transparency in medical decision-making

🧠 Agile Methodology

The project follows Agile methodology, involving:

Iterative development

Continuous feedback

Incremental releases

Frequent testing & improvement

🛠️ Technologies Used
Category	Tools
Language	Python 3.12
Backend	Flask
ML	Scikit-learn, XGBoost
XAI	SHAP
Database	SQLite
Frontend	HTML, CSS, Bootstrap
Visualization	Chart.js
Version Control	Git, GitHub

🎓 Academic Use

MCA Final Year Project

Suitable for:

Machine Learning

Data Analytics

Healthcare AI

Explainable AI

Software Engineering

📌 Future Enhancements

Live patient data integration

Cloud deployment

Deep learning models

Role-based dashboards

Mobile application support



