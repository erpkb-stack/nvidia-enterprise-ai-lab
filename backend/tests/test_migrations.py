import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, inspect

from app.database import Base

BACKEND = Path(__file__).resolve().parents[1]


def test_alembic_upgrade_matches_models():
    db = Path(tempfile.mkdtemp()) / "mig.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db}"}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=BACKEND, env=env,
                   check=True, capture_output=True)
    tables = set(inspect(create_engine(f"sqlite:///{db}")).get_table_names())
    assert set(Base.metadata.tables) <= tables
