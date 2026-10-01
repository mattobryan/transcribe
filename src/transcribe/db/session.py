"""Engine, sessions and migrations (SQLite in WAL mode with foreign keys on)."""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

MIGRATIONS = Path(__file__).parent / "migrations"


def create_engine_for(path: str) -> Engine:
    """Engine for a SQLite file (``:memory:`` for tests)."""
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            cursor.execute("PRAGMA journal_mode = WAL")
        cursor.close()

    return engine


def session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def migrate(path: str, revision: str = "head") -> None:
    """Apply Alembic migrations to the SQLite file at ``path``."""
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, revision)
