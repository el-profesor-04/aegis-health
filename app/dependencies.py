import os
from pathlib import Path

from app.sqlite_graph import SQLiteHealthGraph


_graph = None


def default_db_path():
    return Path(os.getenv("AEGIS_DB_PATH", "data/aegis.sqlite3"))


def get_graph():
    global _graph
    if _graph is None:
        _graph = SQLiteHealthGraph(default_db_path())
    return _graph
