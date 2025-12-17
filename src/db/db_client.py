import sqlite3
import json
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(__file__).parent.parent.parent / "app_data.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    username TEXT,
    input TEXT,
    results TEXT
);
"""

class DBClient:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB)
        self._ensure_db()

    # ✅ ADD THIS METHOD (MISSING EARLIER)
    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure_db(self):
        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute(CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def insert_record(self, timestamp, input_json, results_json, username=None):
        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO predictions (timestamp, username, input, results)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, username, input_json, results_json)
            )
            conn.commit()
        finally:
            conn.close()

    # ✅ USED BY ADMIN DASHBOARD
    def fetch_all(self):
        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute(
                "SELECT id, timestamp, username, input, results FROM predictions ORDER BY id DESC"
            )
            rows = c.fetchall()

            records = []
            for r in rows:
                records.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "username": r[2],
                    "input": json.loads(r[3]) if r[3] else {},
                    "results": json.loads(r[4]) if r[4] else {}
                })
            return records
        finally:
            conn.close()

    # ✅ USED BY PDF DOWNLOAD
    def fetch_recent(self, limit=1):
        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute(
                """
                SELECT id, timestamp, username, input, results
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = c.fetchall()

            records = []
            for r in rows:
                records.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "username": r[2],
                    "input": json.loads(r[3]) if r[3] else {},
                    "results": json.loads(r[4]) if r[4] else {}
                })
            return records
        finally:
            conn.close()
