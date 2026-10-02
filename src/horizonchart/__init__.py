"""Clean horizon-view sky charts built on starplot."""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("horizonchart")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"

SHARED_DATA_PATH = Path("/var/data")


def _data_path() -> Path:
    """Where catalogs, ephemerides and caches go: $STARPLOT_DATA_PATH if set,
    else /var/data if it exists and is writable, else the per-user cache
    folder for this platform."""
    if os.environ.get("STARPLOT_DATA_PATH"):
        return Path(os.environ["STARPLOT_DATA_PATH"])
    if SHARED_DATA_PATH.is_dir() and os.access(SHARED_DATA_PATH, os.W_OK):
        return SHARED_DATA_PATH
    from platformdirs import user_cache_path

    return user_cache_path("horizonchart", appauthor=False)


DATA_PATH = _data_path()
DATA_PATH.mkdir(parents=True, exist_ok=True)
# Starplot fixes its data path when it's first imported, so this has to happen
# before anything imports starplot (importing horizonchart does it first)
os.environ["STARPLOT_DATA_PATH"] = str(DATA_PATH)
