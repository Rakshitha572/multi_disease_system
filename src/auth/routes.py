# src/auth/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from src.models.user import db, User
from datetime import datetime

auth_bp = Blueprint("auth", __name__, template_folder="../webapp/templates")

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")

        if not email or not password:
            flash("Email and password required.", "danger")
            return render_template("register.html", email=email, name=name)

        if password != password2:
            flash("Passwords do not match.", "danger")
            return render_template("register.html", email=email, name=name)

        # check if user exists
        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("auth.login"))

        # create user
        user = User(email=email, name=name, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        session["user"] = user.email
        flash("Registration successful — logged in.", "success")
        next_url = request.args.get("next") or url_for("index")
        return redirect(next_url)
    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session["user"] = user.email
            flash("Logged in successfully.", "success")
            nxt = request.args.get("next") or url_for("index")
            return redirect(nxt)
        flash("Invalid email or password.", "danger")
        return render_template("login.html", email=email)
    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))
