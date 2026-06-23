import sqlite3
from pathlib import Path
from contextlib import contextmanager


DB_PATH = Path(__file__).parent.parent / "data" / "survey.db"


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_conn(path: Path = DB_PATH):
    conn = get_db(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with db_conn(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                site        TEXT,
                date        TEXT,
                folder_path TEXT,
                excel_path  TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS corals (
                id             INTEGER PRIMARY KEY,
                session_id     INTEGER NOT NULL REFERENCES sessions(id),
                genotype_id    TEXT,
                species        TEXT,
                depth_m        REAL,
                photo_a_path   TEXT,
                photo_b_path   TEXT,
                best_photo_path TEXT,
                exif_time      TEXT,
                status         TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS measurements (
                id                INTEGER PRIMARY KEY,
                coral_id          INTEGER NOT NULL REFERENCES corals(id),
                l_mean            REAL,
                a_mean            REAL,
                b_mean            REAL,
                area_px           INTEGER,
                scale_mm_px       REAL,
                area_cm2          REAL,
                whibal_correction REAL,
                quality_flag      TEXT DEFAULT 'ok',
                notes             TEXT,
                mask_path         TEXT,
                measured_at       TEXT DEFAULT (datetime('now'))
            );
        """)


def list_sessions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT s.*,
               COUNT(c.id) AS total,
               SUM(c.status = 'confirmed') AS confirmed,
               SUM(c.status = 'skipped')  AS skipped
        FROM sessions s
        LEFT JOIN corals c ON c.session_id = s.id
        GROUP BY s.id
        ORDER BY s.created_at DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_session(conn: sqlite3.Connection, session_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def create_session(conn: sqlite3.Connection, name: str, site: str, date: str,
                   folder_path: str, excel_path: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO sessions (name, site, date, folder_path, excel_path) VALUES (?,?,?,?,?)",
        (name, site, date, folder_path, excel_path)
    )
    return cur.lastrowid


def insert_coral(conn: sqlite3.Connection, session_id: int, **kwargs) -> int:
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    cur = conn.execute(
        f"INSERT INTO corals (session_id, {cols}) VALUES (?, {placeholders})",
        (session_id, *kwargs.values())
    )
    return cur.lastrowid


def get_coral(conn: sqlite3.Connection, coral_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM corals WHERE id = ?", (coral_id,)).fetchone()
    return dict(row) if row else None


def list_corals(conn: sqlite3.Connection, session_id: int,
                status: str | None = None) -> list[dict]:
    if status:
        rows = conn.execute(
            "SELECT * FROM corals WHERE session_id = ? AND status = ? ORDER BY id",
            (session_id, status)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM corals WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_coral_status(conn: sqlite3.Connection, coral_id: int, status: str):
    conn.execute("UPDATE corals SET status = ? WHERE id = ?", (status, coral_id))


def save_measurement(conn: sqlite3.Connection, coral_id: int, **kwargs) -> int:
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    cur = conn.execute(
        f"INSERT INTO measurements (coral_id, {cols}) VALUES (?, {placeholders})",
        (coral_id, *kwargs.values())
    )
    return cur.lastrowid


def list_confirmed_pairs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT c.best_photo_path, m.mask_path, c.species, c.genotype_id
        FROM measurements m
        JOIN corals c ON c.id = m.coral_id
        WHERE c.status = 'confirmed' AND m.mask_path IS NOT NULL
        ORDER BY m.measured_at
    """).fetchall()
    return [dict(r) for r in rows]
