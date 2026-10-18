"""Text and HTML summaries of fitted models, in the layout of statsmodels.

A :class:`Summary` is a sequence of blocks: a title, header blocks of label and
value pairs, tables, and notes. Printing it gives fixed-width text; in a notebook
it renders as HTML. The tables stay available as DataFrames.
"""

from __future__ import annotations

import html
import numbers
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Summary", "format_value"]

WIDTH = 78

Format = str | Callable[[object], str]


def format_value(value, spec: Format = "{:.4f}") -> str:
    """Format one cell: missing values are blank, integers print as integers."""
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if not np.isfinite(value):
        return "" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return spec(value) if callable(spec) else spec.format(value)


@dataclass
class _Table:
    frame: pd.DataFrame
    title: str | None
    formats: dict
    default: Format


@dataclass
class Summary:
    """A summary made of a title, header blocks, tables and notes.

    Examples
    --------
    >>> s = Summary("Example")
    >>> s.add_header([("No. observations:", 600)], [("Log-likelihood:", 123.4)])
    >>> s.add_table(pd.DataFrame({"coef": [1.0]}, index=["const"]))
    >>> print(s)                                              # doctest: +SKIP
    """

    title: str
    blocks: list = field(default_factory=list)

    # ---------------------------------------------------------------- building

    def add_header(
        self,
        left: Sequence[tuple[str, object]],
        right: Sequence[tuple[str, object]] = (),
        spec: Format = "{:.4f}",
    ) -> Summary:
        """Add a block of label and value pairs in two columns."""
        self.blocks.append(("header", list(left), list(right), spec))
        return self

    def add_table(
        self,
        frame: pd.DataFrame,
        title: str | None = None,
        formats: dict | None = None,
        default: Format = "{:.4f}",
    ) -> Summary:
        """Add a table. ``formats`` maps column names to format specs."""
        self.blocks.append(
            ("table", _Table(frame, title, dict(formats or {}), default))
        )
        return self

    def add_notes(self, notes: Sequence[str]) -> Summary:
        """Add numbered notes at the end."""
        self.blocks.append(("notes", list(notes)))
        return self

    @property
    def tables(self) -> list[pd.DataFrame]:
        """The tables of the summary, as DataFrames."""
        return [b[1].frame for b in self.blocks if b[0] == "table"]

    # ------------------------------------------------------------------- text

    def as_text(self) -> str:
        rendered = [self._render(b) for b in self.blocks]
        width = max([WIDTH] + [w for _, w in rendered])
        lines = [self.title.center(width).rstrip(), "=" * width]
        for i, (block_lines, _) in enumerate(rendered):
            kind = self.blocks[i][0]
            lines.extend(_widen(block_lines, width))
            if kind != "notes":
                lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.as_text()

    def __repr__(self) -> str:
        return self.as_text()

    def _render(self, block) -> tuple[list[str], int]:
        kind = block[0]
        if kind == "header":
            return _render_header(block[1], block[2], block[3]), WIDTH
        if kind == "table":
            return _render_table(block[1])
        lines = []
        for i, note in enumerate(block[1], start=1):
            prefix = f"[{i}] "
            lines.extend(
                textwrap.wrap(
                    note,
                    width=WIDTH,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix),
                )
            )
        return lines, 0

    # ------------------------------------------------------------------- html

    def as_html(self) -> str:
        parts = [f"<div class='rwls-summary'><h4>{html.escape(self.title)}</h4>"]
        for block in self.blocks:
            kind = block[0]
            if kind == "header":
                rows = []
                pairs = list(zip(block[1], block[2], strict=False)) + [
                    (p, None) for p in block[1][len(block[2]) :]
                ]
                for left, right in pairs:
                    cells = [left] if right is None else [left, right]
                    tds = "".join(
                        f"<th>{html.escape(str(k))}</th>"
                        f"<td>{html.escape(format_value(v, block[3]))}</td>"
                        for k, v in cells
                    )
                    rows.append(f"<tr>{tds}</tr>")
                parts.append(f"<table class='rwls-header'>{''.join(rows)}</table>")
            elif kind == "table":
                parts.append(_table_html(block[1]))
            else:
                items = "".join(f"<li>{html.escape(n)}</li>" for n in block[1])
                parts.append(f"<ol class='rwls-notes'>{items}</ol>")
        parts.append("</div>")
        return "".join(parts)

    def _repr_html_(self) -> str:
        return self.as_html()


# ------------------------------------------------------------------ rendering


def _render_header(left, right, spec) -> list[str]:
    half = WIDTH // 2
    gutter = 3
    lines = []
    for i in range(max(len(left), len(right))):
        cells = []
        for column, width in ((left, half - gutter), (right, WIDTH - half)):
            if i < len(column):
                label, value = column[i]
                text = format_value(value, spec)
                cells.append(f"{label}{text:>{width - len(label)}}")
            else:
                cells.append(" " * width)
        lines.append((cells[0] + " " * gutter + cells[1]).rstrip())
    return lines


def _cells(table: _Table) -> tuple[list[str], list[str], list[list[str]]]:
    frame = table.frame
    headers = [str(c) for c in frame.columns]
    labels = [str(i) for i in frame.index]
    body = []
    for _, row in frame.iterrows():
        body.append(
            [
                format_value(row[c], table.formats.get(c, table.default))
                for c in frame.columns
            ]
        )
    return headers, labels, body


def _render_table(table: _Table) -> tuple[list[str], int]:
    headers, labels, body = _cells(table)
    corner = str(table.frame.index.name or "")
    label_w = max([len(corner)] + [len(x) for x in labels])
    widths = [
        max([len(h)] + [len(r[j]) for r in body]) + 2 for j, h in enumerate(headers)
    ]
    widths = [max(w, 10) for w in widths]
    natural = label_w + sum(widths)
    # Like statsmodels, the label column absorbs any space left in the line.
    label_w += max(WIDTH - natural, 0)
    width = label_w + sum(widths)

    lines = []
    if table.title:
        lines.append(table.title.center(width).rstrip())
        lines.append("-" * width)
    lines.append(
        f"{corner:<{label_w}}"
        + "".join(f"{h:>{w}}" for h, w in zip(headers, widths, strict=True))
    )
    lines.append("-" * width)
    for label, row in zip(labels, body, strict=True):
        lines.append(
            f"{label:<{label_w}}"
            + "".join(f"{c:>{w}}" for c, w in zip(row, widths, strict=True))
        )
    return lines, width


def _widen(lines: list[str], width: int) -> list[str]:
    """Stretch rules to the summary's full width."""
    out = []
    for line in lines:
        if line and set(line) <= {"-"}:
            out.append("-" * width)
        else:
            out.append(line)
    return out


def _table_html(table: _Table) -> str:
    headers, labels, body = _cells(table)
    corner = html.escape(str(table.frame.index.name or ""))
    head = f"<th>{corner}</th>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th>"
        + "".join(f"<td>{html.escape(c)}</td>" for c in row)
        + "</tr>"
        for label, row in zip(labels, body, strict=True)
    )
    caption = f"<caption>{html.escape(table.title)}</caption>" if table.title else ""
    return (
        f"<table class='rwls-table'>{caption}<thead><tr>{head}</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
