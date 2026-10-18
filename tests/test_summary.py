import numpy as np
import pandas as pd

from rwls.summary import WIDTH, Summary, format_value


def _example():
    s = Summary("Example Results")
    s.add_header(
        [("Dep. variable:", "AAPL"), ("No. observations:", 600)],
        [("Log-likelihood:", 1234.5678), ("No. regimes:", 3)],
    )
    coefs = pd.DataFrame(
        {"coef": [0.0012, 1.04], "std err": [0.0008, 0.041], "P>|t|": [0.134, 0.0]},
        index=pd.Index(["const", "Mkt-RF"], name="Regime 1"),
    )
    s.add_table(coefs, formats={"P>|t|": "{:.3f}"})
    s.add_notes(["Standard errors are sandwich estimates."])
    return s


def test_format_value():
    assert format_value(1.23456) == "1.2346"
    assert format_value(1.23456, "{:.2f}") == "1.23"
    assert format_value(600) == "600"
    assert format_value(np.int64(7)) == "7"
    assert format_value(np.nan) == ""
    assert format_value(None) == ""
    assert format_value(np.inf) == "inf"
    assert format_value("AAPL") == "AAPL"
    assert format_value(True) == "True"
    assert format_value(0.5, lambda v: f"{v:.0%}") == "50%"


def test_text_layout_follows_statsmodels():
    lines = str(_example()).splitlines()
    assert lines[0].strip() == "Example Results"
    assert lines[1] == "=" * WIDTH
    # Header pairs: label at the left of each half, value at its right edge.
    assert lines[2].startswith("Dep. variable:")
    assert lines[2][: WIDTH // 2].rstrip().endswith("AAPL")
    assert lines[2].endswith("1234.5678")
    assert lines[3][: WIDTH // 2].rstrip().endswith("600")
    assert lines[4] == "=" * WIDTH
    # Table: header row, rule, one row per index entry, numbers right-aligned.
    assert lines[5].startswith("Regime 1")
    assert lines[5].endswith("P>|t|")
    assert lines[6] == "-" * WIDTH
    assert lines[7].startswith("const") and lines[7].endswith("0.134")
    assert lines[8].startswith("Mkt-RF") and lines[8].endswith("0.000")
    assert lines[9] == "=" * WIDTH
    assert lines[10] == "[1] Standard errors are sandwich estimates."
    assert all(len(line) <= WIDTH for line in lines)


def test_wide_tables_widen_the_rules():
    s = Summary("Wide")
    wide = pd.DataFrame(np.ones((1, 9)), columns=[f"Regime {i}" for i in range(1, 10)])
    s.add_table(wide)
    lines = str(s).splitlines()
    width = len(lines[1])
    assert width > WIDTH
    assert lines[-1] == "=" * width
    assert all(len(line) <= width for line in lines)


def test_table_titles_and_missing_values():
    s = Summary("T")
    frame = pd.DataFrame({"a": [1.0, np.nan]}, index=["x", "y"])
    s.add_table(frame, title="Factor means")
    text = str(s)
    assert "Factor means" in text
    assert text.splitlines()[-2].rstrip() == "y"


def test_html_and_tables():
    s = _example()
    out = s._repr_html_()
    assert out.startswith("<div class='rwls-summary'>")
    assert "<h4>Example Results</h4>" in out
    assert "<td>1.0400</td>" in out
    assert "P&gt;|t|" in out
    assert len(s.tables) == 1
    assert list(s.tables[0].index) == ["const", "Mkt-RF"]


def test_long_notes_are_wrapped():
    s = Summary("T")
    s.add_notes(["word " * 40])
    lines = str(s).splitlines()[2:]
    assert len(lines) > 1
    assert lines[0].startswith("[1] word")
    assert lines[1].startswith("    word")
    assert all(len(line) <= WIDTH for line in lines)
