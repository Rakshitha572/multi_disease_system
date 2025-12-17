# run_web.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.webapp.app import app

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
