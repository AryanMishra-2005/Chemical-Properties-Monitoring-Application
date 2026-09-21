"""
init_db.py
----------
Creates (or resets) the SQLite database and seeds it with critical constants
for the four supported chemicals.

Data sources
  Tc, Pc  : NIST / Perry's Chemical Engineers' Handbook
  omega   : Standard Pitzer acentric factor tables
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("chemicals.db")

# ------------------------------------------------------------------
# Seed data
# ------------------------------------------------------------------
CHEMICALS = [
    #  name       CAS           Tc (K)   Pc (Pa)        omega    formula
    ("Methane",  "74-82-8",    190.56,  4_599_200.0,  0.01142,  "CH4"),
    ("Ethane",   "74-84-0",    305.32,  4_871_800.0,  0.09949,  "C2H6"),
    ("Propane",  "74-98-6",    369.83,  4_247_200.0,  0.15228,  "C3H8"),
    ("Water",    "7732-18-5",  647.10,  22_064_000.0, 0.34510,  "H2O"),
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chemicals (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL UNIQUE,
    cas       TEXT    NOT NULL,
    Tc        REAL    NOT NULL,   -- Critical temperature  [K]
    Pc        REAL    NOT NULL,   -- Critical pressure     [Pa]
    omega     REAL    NOT NULL,   -- Pitzer acentric factor [-]
    formula   TEXT    NOT NULL
);
"""

INSERT_ROW = """
INSERT OR REPLACE INTO chemicals (name, cas, Tc, Pc, omega, formula)
VALUES (?, ?, ?, ?, ?, ?);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(CREATE_TABLE)
    cur.executemany(INSERT_ROW, CHEMICALS)
    con.commit()
    con.close()
    print(f"Database ready at '{db_path}' with {len(CHEMICALS)} chemicals.")


if __name__ == "__main__":
    init_db()
