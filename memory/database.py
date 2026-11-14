
import sqlite3
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "opportunitylens.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    """Create tables and safely migrate older database schemas."""

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_key TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL DEFAULT 'anonymous',
                opportunity TEXT DEFAULT 'Unknown',
                result TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT ''
            )
        """)

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(assessments)"
            ).fetchall()
        }

        if "user_key" not in columns:
            conn.execute("""
                ALTER TABLE assessments
                ADD COLUMN user_key TEXT DEFAULT 'anonymous'
            """)

        if "opportunity" not in columns:
            conn.execute("""
                ALTER TABLE assessments
                ADD COLUMN opportunity TEXT DEFAULT 'Unknown'
            """)

        if "result" not in columns:
            conn.execute("""
                ALTER TABLE assessments
                ADD COLUMN result TEXT DEFAULT '{}'
            """)

        if "created_at" not in columns:
            conn.execute("""
                ALTER TABLE assessments
                ADD COLUMN created_at TEXT DEFAULT ''
            """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_assessments_user_id
            ON assessments(user_key, id)
        """)


# ============================================================
# PROFILE FUNCTIONS
# ============================================================

def save_profile(user_key, profile):
    """Save or update an applicant profile."""

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO profiles
            (user_key, profile, updated_at)
            VALUES (?, ?, ?)
        """, (
            str(user_key),
            json.dumps(
                profile,
                ensure_ascii=False,
                default=str
            ),
            datetime.now().isoformat(timespec="seconds")
        ))


def load_profile(user_key):
    """Load a saved applicant profile."""

    with get_connection() as conn:
        row = conn.execute("""
            SELECT profile
            FROM profiles
            WHERE user_key = ?
        """, (str(user_key),)).fetchone()

    if row is None:
        return {}

    try:
        profile = json.loads(row["profile"])
        return profile if isinstance(profile, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ============================================================
# SAVE ASSESSMENT
# ============================================================

def save_assessment(user_key, opportunity, result=None):
    """
    Save an assessment.

    Supports:
        save_assessment(user_key, result)
        save_assessment(user_key, opportunity, result)
    """

    if result is None:
        result = opportunity
        opportunity_name = "Unknown"

        if isinstance(result, dict):
            opportunity_data = result.get("opportunity", {})

            if isinstance(opportunity_data, dict):
                opportunity_name = (
                    opportunity_data.get("title")
                    or opportunity_data.get("name")
                    or "Unknown"
                )

    elif isinstance(opportunity, dict):
        opportunity_name = (
            opportunity.get("name")
            or opportunity.get("title")
            or "Unknown"
        )

    else:
        opportunity_name = str(opportunity)

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO assessments
            (user_key, opportunity, result, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            str(user_key),
            str(opportunity_name),
            json.dumps(
                result,
                ensure_ascii=False,
                default=str
            ),
            datetime.now().isoformat(timespec="seconds")
        ))

        return cursor.lastrowid


# ============================================================
# GET RECENT ASSESSMENTS
# ============================================================

def recent_assessments(user_key, limit=50):
    """Return assessment summaries for a profile."""

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, opportunity, result, created_at
            FROM assessments
            WHERE user_key = ?
            ORDER BY id DESC
            LIMIT ?
        """, (
            str(user_key),
            int(limit)
        )).fetchall()

    history = []

    for row in rows:

        try:
            result = json.loads(row["result"])
        except (json.JSONDecodeError, TypeError):
            result = {}

        if not isinstance(result, dict):
            result = {}

        eligibility = result.get("eligibility", {})

        if not isinstance(eligibility, dict):
            eligibility = {}

        history.append({
            "ID": row["id"],
            "Opportunity": row["opportunity"] or "Unknown",
            "Date": row["created_at"] or "Unknown",
            "Result": eligibility.get("status", "Unknown")
        })

    return history


# ============================================================
# GET ONE COMPLETE ASSESSMENT
# ============================================================

def get_assessment(user_key, assessment_id):
    """
    Retrieve the complete original assessment report.

    The user_key condition ensures that one profile cannot
    access another profile's assessment.
    """

    with get_connection() as conn:
        row = conn.execute("""
            SELECT id, user_key, opportunity, result, created_at
            FROM assessments
            WHERE user_key = ? AND id = ?
        """, (
            str(user_key),
            int(assessment_id)
        )).fetchone()

    if row is None:
        return None

    try:
        result = json.loads(row["result"])
    except (json.JSONDecodeError, TypeError):
        result = {}

    if not isinstance(result, dict):
        result = {"report": str(result)}

    return {
        "id": row["id"],
        "user_key": row["user_key"],
        "opportunity": row["opportunity"] or "Unknown",
        "result": result,
        "created_at": row["created_at"] or "Unknown"
    }


# ============================================================
# DELETE ONE ASSESSMENT
# ============================================================

def delete_assessment(user_key, assessment_id):
    """Delete one assessment belonging to the selected profile."""

    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM assessments
            WHERE user_key = ? AND id = ?
        """, (
            str(user_key),
            int(assessment_id)
        ))

        return cursor.rowcount > 0


# ============================================================
# DELETE ALL ASSESSMENTS FOR ONE PROFILE
# ============================================================

def delete_all_assessments(user_key):
    """
    Delete all assessment history belonging to one profile.

    Applicant profiles and other users' assessments remain intact.
    """

    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM assessments
            WHERE user_key = ?
        """, (str(user_key),))

        return cursor.rowcount


# ============================================================
# COUNT ASSESSMENTS
# ============================================================

def count_assessments(user_key):
    """Return the number of saved assessments for a profile."""

    with get_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total
            FROM assessments
            WHERE user_key = ?
        """, (str(user_key),)).fetchone()

    return row["total"] if row else 0
