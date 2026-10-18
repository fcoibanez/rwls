"""Factor returns from the Kenneth R. French Data Library.

Downloads the monthly research factors behind the CAPM, the Fama-French three- and
five-factor models, and their extensions with momentum. Any other source can be
used instead: the estimators accept factor returns as a plain DataFrame.

French revises the factors when the underlying CRSP data are updated, so results
depend on the date of download. Each result records the CRSP vintage and the date
the files were retrieved in ``DataFrame.attrs``. Passing ``cache_dir`` stores the
downloaded files and reuses them on later calls, which pins the vintage.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["MODELS", "load_factors"]

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

_THREE = "F-F_Research_Data_Factors"
_FIVE = "F-F_Research_Data_5_Factors_2x3"
_MOMENTUM = "F-F_Momentum_Factor"

#: Factor columns of each model and the file each column is taken from. The size
#: and value factors differ between the three- and five-factor files, so each
#: model uses its own file throughout. The risk-free rate comes from the model's
#: main file.
MODELS: dict[str, list[tuple[str, list[str]]]] = {
    "capm": [(_THREE, ["Mkt-RF"])],
    "ff3": [(_THREE, ["Mkt-RF", "SMB", "HML"])],
    "ff3+mom": [(_THREE, ["Mkt-RF", "SMB", "HML"]), (_MOMENTUM, ["Mom"])],
    "ff5": [(_FIVE, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])],
    "ff5+mom": [(_FIVE, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]), (_MOMENTUM, ["Mom"])],
}

_ALIASES = {"carhart": "ff3+mom", "ff4": "ff3+mom", "ff6": "ff5+mom"}

#: Column names used by WRDS for the same series.
WRDS_NAMES = {
    "Mkt-RF": "mktrf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "Mom": "umd",
    "RF": "rf",
}

_MISSING = (-99.99, -999.0)
_ROW = re.compile(r"^\s*(\d{6})\s*,(.*)$")
_VINTAGE = re.compile(r"created using the (\d{6}) CRSP database", re.IGNORECASE)


def load_factors(
    model: str = "ff5+mom",
    start=None,
    end=None,
    names: str | dict[str, str] = "french",
    cache_dir: str | os.PathLike | None = None,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Monthly factor returns and the risk-free rate, in decimals.

    Parameters
    ----------
    model : str
        One of ``"capm"``, ``"ff3"``, ``"ff3+mom"``, ``"ff5"`` and ``"ff5+mom"``.
        ``"carhart"`` and ``"ff4"`` are accepted for ``"ff3+mom"``, and ``"ff6"``
        for ``"ff5+mom"``.
    start, end : str, datetime-like or None
        First and last month to keep, inclusive. Any value pandas can parse as a
        date, such as ``"1963-07"``.
    names : {"french", "wrds"} or dict
        Column names. ``"french"`` keeps the library's names (``Mkt-RF``, ``SMB``,
        ``HML``, ``RMW``, ``CMA``, ``Mom``, ``RF``); ``"wrds"`` uses ``mktrf``,
        ``smb``, ``hml``, ``rmw``, ``cma``, ``umd``, ``rf``. A dict maps French's
        names to any others; names it does not mention are kept.
    cache_dir : path-like or None
        Directory in which downloaded files are stored. Files already there are
        used instead of downloading, unless ``refresh`` is True.
    refresh : bool
        Download again even if the files are in ``cache_dir``.
    timeout : float
        Seconds to wait for each download.

    Returns
    -------
    pandas.DataFrame
        One row per month, indexed by month-end dates, with the model's factors
        followed by ``RF``. Months missing from any file used are dropped.
        ``attrs`` records the model, the source files, the CRSP vintage and the
        retrieval date.
    """
    key = _ALIASES.get(model.lower(), model.lower())
    if key not in MODELS:
        known = ", ".join(sorted(MODELS) + sorted(_ALIASES))
        raise ValueError(f"unknown model {model!r}; choose one of: {known}")

    parts, files, vintages, retrieved = [], [], set(), []
    for i, (file, columns) in enumerate(MODELS[key]):
        raw, when = _fetch(file, cache_dir, refresh, timeout)
        text = _read_csv_from_zip(raw)
        table = _parse_monthly(text)
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise ValueError(f"{file} has no column(s) {missing}")
        keep = columns + (["RF"] if i == 0 else [])
        parts.append(table[keep])
        files.append(file)
        retrieved.append(when)
        match = _VINTAGE.search(text)
        if match:
            vintages.add(match.group(1))

    out = pd.concat(parts, axis=1, join="inner").dropna(how="any")
    out = out[[c for c in out.columns if c != "RF"] + ["RF"]]

    if start is not None:
        out = out.loc[pd.Timestamp(start) + pd.offsets.MonthEnd(0) :]
    if end is not None:
        out = out.loc[: pd.Timestamp(end) + pd.offsets.MonthEnd(0)]

    out = out.rename(columns=_name_map(names))
    out.attrs = {
        "source": "Kenneth R. French Data Library",
        "model": key,
        "files": files,
        "crsp_vintage": sorted(vintages),
        "retrieved": min(retrieved).isoformat(),
    }
    return out


def _name_map(names: str | dict[str, str]) -> dict[str, str]:
    if isinstance(names, dict):
        return dict(names)
    if names == "french":
        return {}
    if names == "wrds":
        return dict(WRDS_NAMES)
    raise ValueError("names must be 'french', 'wrds' or a dict")


def _fetch(
    file: str, cache_dir: str | os.PathLike | None, refresh: bool, timeout: float
) -> tuple[bytes, dt.date]:
    """The zipped CSV for ``file`` and the date it was retrieved."""
    name = f"{file}_CSV.zip"
    if cache_dir is not None:
        path = Path(cache_dir) / name
        if path.exists() and not refresh:
            when = dt.date.fromtimestamp(path.stat().st_mtime)
            return path.read_bytes(), when

    request = urllib.request.Request(
        BASE_URL + name, headers={"User-Agent": "rwls (Python package)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()

    if cache_dir is not None:
        path = Path(cache_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return raw, dt.date.today()


def _read_csv_from_zip(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV file in the archive, found {members}")
        return archive.read(members[0]).decode("latin-1")


def _parse_monthly(text: str) -> pd.DataFrame:
    """The monthly section of a French CSV file, in decimals.

    The monthly section starts at the first header line, which begins with a
    comma, and runs while lines start with a six-digit year and month. Later
    sections (annual factors) are ignored.
    """
    lines = text.splitlines()
    try:
        head = next(i for i, line in enumerate(lines) if line.lstrip().startswith(","))
    except StopIteration:
        raise ValueError("no header line found in the factor file") from None

    columns = [c.strip() for c in lines[head].split(",")[1:]]
    dates, rows = [], []
    for line in lines[head + 1 :]:
        match = _ROW.match(line)
        if not match:
            break
        values = [float(v) for v in match.group(2).split(",")]
        if len(values) != len(columns):
            raise ValueError(f"row {match.group(1)} has {len(values)} values")
        dates.append(match.group(1))
        rows.append(values)
    if not rows:
        raise ValueError("no monthly rows found in the factor file")

    data = np.array(rows, dtype=float)
    data[np.isin(data, _MISSING)] = np.nan
    index = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    frame = pd.DataFrame(data / 100.0, index=pd.DatetimeIndex(index, name="date"))
    frame.columns = columns
    return frame
