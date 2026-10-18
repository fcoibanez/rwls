"""Tests for the French data library reader.

The offline tests use synthetic files in the library's layout, written to a cache
directory so that no download takes place. One test downloads the real files and
is marked ``network``.
"""

import zipfile

import numpy as np
import pandas as pd
import pytest

from rwls.factors import MODELS, load_factors

HEADER = (
    "This file was created using the 202608 CRSP database.\r\n"
    "The 1-month TBill rate data are from a source.\r\n"
    "\r\n"
)


def _annual(columns):
    return (
        "\r\n Annual Factors: January-December \r\n,"
        + ",".join(columns)
        + "\r\n  1999,  1.00\r\n"
    )


def _write(cache, file, columns, rows):
    body = HEADER + "," + ",".join(columns) + "\r\n"
    body += "".join(
        f"{d},{','.join(f'{v:8.2f}' for v in vals)}\r\n" for d, vals in rows
    )
    body += _annual(columns) + "\r\nCopyright Eugene F. Fama and Kenneth R. French\r\n"
    with zipfile.ZipFile(cache / f"{file}_CSV.zip", "w") as archive:
        archive.writestr(f"{file}.csv", body)


@pytest.fixture
def cache(tmp_path):
    months = ["202001", "202002", "202003"]
    _write(
        tmp_path,
        "F-F_Research_Data_Factors",
        ["Mkt-RF", "SMB", "HML", "RF"],
        [(m, [1.0 + i, 2.0, 3.0, 0.1]) for i, m in enumerate(months)],
    )
    _write(
        tmp_path,
        "F-F_Research_Data_5_Factors_2x3",
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        [(m, [1.0 + i, 2.5, 3.0, 4.0, 5.0, 0.1]) for i, m in enumerate(months)],
    )
    _write(
        tmp_path,
        "F-F_Momentum_Factor",
        ["Mom   "],
        [("202002", [6.0]), ("202003", [-99.99]), ("202004", [7.0])],
    )
    return tmp_path


def test_values_are_converted_to_decimals(cache):
    out = load_factors("ff3", cache_dir=cache)
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert out["Mkt-RF"].tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert out["RF"].tolist() == pytest.approx([0.001] * 3)


def test_index_is_month_end_dates(cache):
    out = load_factors("capm", cache_dir=cache)
    assert list(out.columns) == ["Mkt-RF", "RF"]
    assert out.index.name == "date"
    assert list(out.index) == list(
        pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    )


def test_each_model_uses_its_own_file(cache):
    assert load_factors("ff3", cache_dir=cache)["SMB"].iloc[0] == pytest.approx(0.02)
    assert load_factors("ff5", cache_dir=cache)["SMB"].iloc[0] == pytest.approx(0.025)


def test_momentum_is_joined_and_missing_months_are_dropped(cache):
    out = load_factors("ff5+mom", cache_dir=cache)
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    # Momentum starts in February, March is coded missing and April has no other
    # factors, so only February survives.
    assert list(out.index) == [pd.Timestamp("2020-02-29")]
    assert out["Mom"].iloc[0] == pytest.approx(0.06)


def test_aliases(cache):
    pd.testing.assert_frame_equal(
        load_factors("carhart", cache_dir=cache),
        load_factors("ff3+mom", cache_dir=cache),
    )
    pd.testing.assert_frame_equal(
        load_factors("FF6", cache_dir=cache), load_factors("ff5+mom", cache_dir=cache)
    )


def test_start_and_end_are_inclusive_months(cache):
    out = load_factors("ff3", start="2020-02", end="2020-02", cache_dir=cache)
    assert list(out.index) == [pd.Timestamp("2020-02-29")]


def test_wrds_names_and_custom_names(cache):
    wrds = load_factors("ff5+mom", names="wrds", cache_dir=cache)
    assert list(wrds.columns) == ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    custom = load_factors("ff3", names={"Mkt-RF": "market"}, cache_dir=cache)
    assert list(custom.columns) == ["market", "SMB", "HML", "RF"]


def test_attrs_record_the_source(cache):
    attrs = load_factors("ff5+mom", cache_dir=cache).attrs
    assert attrs["model"] == "ff5+mom"
    assert attrs["files"] == ["F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor"]
    assert attrs["crsp_vintage"] == ["202608"]
    assert isinstance(attrs["retrieved"], str)


def test_unknown_model_and_names_are_rejected(cache):
    with pytest.raises(ValueError, match="unknown model"):
        load_factors("ff7", cache_dir=cache)
    with pytest.raises(ValueError, match="names"):
        load_factors("ff3", names="crsp", cache_dir=cache)


def test_every_model_is_readable(cache):
    for model in MODELS:
        out = load_factors(model, cache_dir=cache)
        assert out.columns[-1] == "RF"
        assert np.isfinite(out.to_numpy()).all()


@pytest.mark.network
def test_download_from_the_library(tmp_path):
    out = load_factors("ff5+mom", start="1963-07", end="2024-12", cache_dir=tmp_path)
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    assert out.index[0] == pd.Timestamp("1963-07-31")
    assert out.index[-1] == pd.Timestamp("2024-12-31")
    assert len(out) == 738
    # Monthly returns in decimals: the market's volatility is a few percent.
    assert 0.02 < out["Mkt-RF"].std() < 0.08
    assert (tmp_path / "F-F_Momentum_Factor_CSV.zip").exists()
    assert out.attrs["crsp_vintage"]
