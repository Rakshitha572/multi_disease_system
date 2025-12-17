# src/db/migrate_db.py
import sqlite3
from pathlib import Path

DB = Path(__file__).parent.parent / "app_data.db"

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cursor.fetchall()]
    return column in cols

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Ensure table exists first (if not, create fresh table)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input TEXT,
        results TEXT
    );
    """)
    conn.commit()

    # Add timestamp column if missing
    if not column_exists(cur, "predictions", "timestamp"):
        print("Adding 'timestamp' column...")
        cur.execute("ALTER TABLE predictions ADD COLUMN timestamp TEXT;")
        conn.commit()
    else:
        print("'timestamp' column already present.")

    # Add username column if missing
    if not column_exists(cur, "predictions", "username"):
        print("Adding 'username' column...")
        cur.execute("ALTER TABLE predictions ADD COLUMN username TEXT;")
        conn.commit()
    else:
        print("'username' column already present.")

    print("Migration completed.")
    conn.close()

if __name__ == "__main__":
    main()
