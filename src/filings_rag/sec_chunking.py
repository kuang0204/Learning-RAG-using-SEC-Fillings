from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from bs4 import BeautifulSoup, Tag


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class Element:
    element_type: str  # heading | paragraph | list | table
    text: str
    part: Optional[str] = None
    item: Optional[str] = None
    section: Optional[str] = None
    source: str = "html"
    source_anchor: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc: str
    company: str
    filing_type: str
    fiscal_period: str
    filing_date: Optional[str]
    cik: Optional[str]
    accession: Optional[str]
    source_url: Optional[str]
    part: Optional[str]
    item: Optional[str]
    section: Optional[str]
    section_path: list[str]
    element_type: str
    source: str
    source_anchor: Optional[str]
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# NORMALISATION / STRUCTURE PATTERNS
# ============================================================


WHITESPACE_RE = re.compile(r"\s+")
PART_RE = re.compile(r"^\s*PART\s+([IVX]+)\s*\.?\s*$", re.IGNORECASE)
ITEM_RE = re.compile(
    r"^\s*ITEM\s+(\d+[A-Z]?)\s*[\.\-:\u2013\u2014]?\s*(.*?)\s*$",
    re.IGNORECASE,
)
ITEM_TITLE_WORD_RE = re.compile(r"[A-Za-z]{2,}")
CURRENCY_CELL_RE = re.compile(r"^[$€£¥]$")
PERCENT_CELL_RE = re.compile(r"^%$")
NUMERIC_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A genuine column label is short. Anything longer means infer_table_header_rows
# swallowed data rows into the header block, so the collapsed label is really
# table content and must not be repeated on every split piece.
MAX_COLUMN_LABEL_CHARS = 200
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
NUMBERISH_RE = re.compile(
    r"^\(?\$?[-+]?\d[\d,]*(?:\.\d+)?%?\)?$|^[-\u2013\u2014]$"
)

SEMANTIC_TAGS = {
    "p",
    "div",
    "li",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

XBRL_FACT_TAGS = {
    "ix:nonfraction",
    "ix:nonnumeric",
    "ix:fraction",
}

XBRL_HIDDEN_TAGS = {
    "ix:hidden",
    "ix:header",
    "ix:references",
    "ix:resources",
}


# ============================================================
# BASIC HELPERS
# ============================================================


def clean_text(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return WHITESPACE_RE.sub(" ", text).strip()


def normalise_key(text: str) -> str:
    return clean_text(text).casefold()


def safe_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def get_anchor(tag: Tag) -> Optional[str]:
    return tag.get("id") or tag.get("name")


def detect_part(text: str) -> Optional[str]:
    match = PART_RE.match(text)
    if not match:
        return None
    return f"Part {match.group(1).upper()}"


def detect_item(text: str) -> Optional[str]:
    match = ITEM_RE.match(text)
    if not match:
        return None

    number = match.group(1).upper()
    title = clean_text(match.group(2)).strip(" .:-\u2013\u2014")

    if title:
        return f"Item {number} — {title}"
    return f"Item {number}"


def is_titled_item_heading(text: str) -> bool:
    """
    True when a block is a real Item heading (number + a worded title), as
    opposed to a bare running page header ("Item 7") or a multi-item page
    label ("Item 1B, 1C"). Filings such as Microsoft's repeat the bare form
    on every page of a section, so it must never be treated as the heading
    that opens the section.
    """

    match = ITEM_RE.match(text)
    if not match:
        return False

    title = clean_text(match.group(2)).strip(" .:-\u2013\u2014")
    return bool(ITEM_TITLE_WORD_RE.search(title))


def item_number(item_label: str) -> str:
    match = re.match(r"Item\s+(\d+[A-Z]?)", item_label, re.IGNORECASE)
    return match.group(1).upper() if match else item_label.casefold()


def build_section_path(
    part: Optional[str],
    item: Optional[str],
    section: Optional[str],
) -> list[str]:
    return [x for x in (part, item, section) if x]


# ============================================================
# HTML CLEANING
# ============================================================


def remove_noncontent_html(soup: BeautifulSoup) -> None:
    """Remove scripts, styles, hidden iXBRL resources, and CSS-hidden nodes."""

    for tag in soup.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        name = (tag.name or "").lower()
        if name in XBRL_HIDDEN_TAGS:
            tag.decompose()

    for tag in list(soup.find_all(True)):
        style = (tag.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()


def should_process_block(tag: Tag) -> bool:
    """
    Keep semantic leaf blocks while avoiding duplicate extraction from
    container divs. Tables are processed as standalone elements.
    """

    if tag.find_parent("table") is not None:
        return False

    if tag.name == "table":
        return True

    if tag.name in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
        return True

    if tag.name == "div":
        for descendant in tag.find_all(SEMANTIC_TAGS, recursive=True):
            if descendant is not tag:
                return False
        return True

    return False


def style_contains(tag: Tag, needle: str) -> bool:
    return needle in (tag.get("style") or "").replace(" ", "").lower()


def has_bold_signal(tag: Tag) -> bool:
    if tag.name in {"b", "strong"}:
        return True

    style = (tag.get("style") or "").replace(" ", "").lower()
    if "font-weight:bold" in style or "font-weight:700" in style:
        return True

    for child in tag.find_all(["b", "strong"], recursive=True):
        if clean_text(child.get_text(" ", strip=True)):
            return True

    for child in tag.find_all(style=True, recursive=True):
        child_style = (child.get("style") or "").replace(" ", "").lower()
        if "font-weight:bold" in child_style or "font-weight:700" in child_style:
            return True

    return False


def has_center_signal(tag: Tag) -> bool:
    if style_contains(tag, "text-align:center"):
        return True

    align = (tag.get("align") or "").strip().lower()
    if align == "center":
        return True

    for child in tag.find_all(style=True, recursive=True):
        if style_contains(child, "text-align:center"):
            return True

    return False


def is_probable_subsection_heading(tag: Tag, text: str) -> bool:
    """
    Conservative subsection-heading detector.

    We deliberately require both short text and a formatting signal. This
    avoids turning every short sentence in an SEC filing into a section.
    """

    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return True

    if not text or len(text) > 160:
        return False

    words = text.split()
    if len(words) > 22:
        return False

    if text.endswith(('.', ';', '?', '!')) and len(words) > 8:
        return False

    digit_chars = sum(ch.isdigit() for ch in text)
    if digit_chars / max(1, len(text)) > 0.35:
        return False

    return has_bold_signal(tag) or has_center_signal(tag)


def heading_table_text(table: Tag) -> Optional[str]:
    """
    Return the heading text when a <table> holds nothing but a Part/Item
    heading.

    Some filers (Amazon among them) wrap each body Item heading in its own
    small table, splitting the number and the title across adjacent cells.
    Such a table is structure, not data: treating it as a data table loses
    the Item transition and leaves the whole section carrying the previous
    Item and a stale subsection label.

    Whole-table text is what is tested, so a real data table cannot match:
    its concatenated cells never read as just "Item 7. <title>".
    """

    text = clean_text(table.get_text(" ", strip=True))
    if not text or len(text) > 220:
        return None

    if detect_part(text) or is_titled_item_heading(text):
        return text

    return None


def is_structural_heading_candidate(tag: Tag, text: str) -> bool:
    """Restrict PART/ITEM detection to short heading-like blocks."""

    if len(text) > 220:
        return False

    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return True

    if has_bold_signal(tag) or has_center_signal(tag):
        return True

    # SEC filings sometimes use plain leaf divs for true Item headings.
    return len(text.split()) <= 18


# ============================================================
# iXBRL METADATA
# ============================================================


def extract_xbrl_metadata(tag: Tag) -> dict[str, Any]:
    """
    Extract lightweight Inline-XBRL metadata from visible facts contained in
    an element. We keep identifiers, not duplicate fact text.
    """

    concepts: set[str] = set()
    contexts: set[str] = set()
    units: set[str] = set()

    candidates: list[Tag] = []

    if (tag.name or "").lower() in XBRL_FACT_TAGS:
        candidates.append(tag)

    for child in tag.find_all(True, recursive=True):
        if (child.name or "").lower() in XBRL_FACT_TAGS:
            candidates.append(child)

    for fact in candidates:
        if fact.get("name"):
            concepts.add(str(fact.get("name")))
        if fact.get("contextref"):
            contexts.add(str(fact.get("contextref")))
        if fact.get("unitref"):
            units.add(str(fact.get("unitref")))

    out: dict[str, Any] = {}
    if concepts:
        out["xbrl_concepts"] = sorted(concepts)
    if contexts:
        out["xbrl_context_refs"] = sorted(contexts)
    if units:
        out["xbrl_units"] = sorted(units)
    if candidates:
        out["xbrl_fact_count"] = len(candidates)

    return out


# ============================================================
# TABLE EXTRACTION
# ============================================================


def text_looks_numeric(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    return bool(NUMBERISH_RE.match(text))


def table_to_grid(table: Tag) -> tuple[list[list[str]], list[bool]]:
    """
    Convert one HTML table to a rectangular grid while respecting rowspan and
    colspan. Returns (rows, row_has_th_flags).
    """

    rows: list[list[str]] = []
    row_has_th: list[bool] = []

    # col -> (remaining_rows_after_current, value)
    active_rowspans: dict[int, tuple[int, str]] = {}

    trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]

    for tr in trs:
        row: list[str] = []
        has_th = False
        col = 0

        def fill_active_until_free(target_col: int) -> int:
            nonlocal row
            c = target_col
            while c in active_rowspans:
                remaining, value = active_rowspans[c]
                while len(row) <= c:
                    row.append("")
                row[c] = value
                if remaining <= 1:
                    del active_rowspans[c]
                else:
                    active_rowspans[c] = (remaining - 1, value)
                c += 1
            return c

        cells = tr.find_all(["th", "td"], recursive=False)

        for cell in cells:
            col = fill_active_until_free(col)

            value = clean_text(cell.get_text(" ", strip=True))
            colspan = safe_int(cell.get("colspan"), 1)
            rowspan = safe_int(cell.get("rowspan"), 1)
            has_th = has_th or cell.name == "th"

            for offset in range(colspan):
                target = col + offset
                while len(row) <= target:
                    row.append("")
                row[target] = value if offset == 0 else ""
                if rowspan > 1:
                    active_rowspans[target] = (rowspan - 1, value if offset == 0 else "")

            col += colspan

        # Fill any remaining carried rowspan cells to the right.
        if active_rowspans:
            max_col = max(active_rowspans)
            while col <= max_col:
                col = fill_active_until_free(col)
                if col <= max_col and col not in active_rowspans:
                    while len(row) <= col:
                        row.append("")
                    col += 1

        if any(cell.strip() for cell in row):
            rows.append(row)
            row_has_th.append(has_th)

    if not rows:
        return [], []

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    keep_cols = [c for c in range(width) if any(row[c].strip() for row in rows)]
    rows = [[row[c] for c in keep_cols] for row in rows]
    row_has_th = row_has_th

    return rows, row_has_th


def escape_markdown_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_table_rows(rows: list[list[str]]) -> list[str]:
    """Legacy pipe-grid rendering. Retained as the no-usable-header fallback."""

    return [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |"
        for row in rows
    ]


def _numeric_tokens(text: str) -> Counter:
    """Multiset of numeric tokens, used to prove a rendering drops no figures."""

    return Counter(NUMERIC_TOKEN_RE.findall(text))


def rows_text(rows: list[list[str]]) -> str:
    return "\n".join(" ".join(row) for row in rows)


def _column_labels(rows: list[list[str]], header_rows: int) -> tuple[list[str], list[str]]:
    """
    Collapse the header block into one label per column, plus table-wide banners.

    A header row holding a single non-empty cell is a banner spanning the whole
    table ("Year Ended", "($ in millions)"), not a label for the one column it
    happens to sit in. Folding those into that column's label would bloat every
    value in it and label the other columns inconsistently, so they are returned
    separately and emitted once. Rows with two or more non-empty cells are real
    column labels and stack top-to-bottom.
    """

    if header_rows <= 0 or not rows:
        return [], []

    width = max(len(row) for row in rows)
    header_block = rows[:header_rows]

    banners: list[str] = []
    label_rows: list[list[str]] = []

    for row in header_block:
        nonempty = [cell.strip() for cell in row if cell.strip()]
        if len(nonempty) == 1:
            if nonempty[0] not in banners:
                banners.append(nonempty[0])
        elif nonempty:
            label_rows.append(row)

    labels: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in label_rows:
            value = row[col].strip() if col < len(row) else ""
            if value and value not in parts:
                parts.append(value)
        labels.append(clean_text(" ".join(parts)))

    return labels, banners


def _render_data_row(row: list[str], labels: list[str]) -> str:
    """
    Render one data row as "header: value" pairs.

    SEC HTML puts currency symbols and percent signs in their own grid
    columns, so they are merged back onto the adjacent value. A merged cell
    takes the label of whichever column in its span carries one, because the
    symbol frequently sits in the labelled column while the digits sit in an
    unlabelled spacer column.
    """

    cells: list[str] = []
    i = 0
    width = len(row)

    def next_nonempty(start: int) -> int:
        j = start
        while j < width and not row[j].strip():
            j += 1
        return j

    while i < width:
        value = row[i].strip()
        if not value:
            i += 1
            continue

        span_start = i

        if CURRENCY_CELL_RE.match(value):
            j = next_nonempty(i + 1)
            if j < width:
                following = row[j].strip()
                if not CURRENCY_CELL_RE.match(following) and not PERCENT_CELL_RE.match(
                    following
                ):
                    value += following
                    i = j

        j = next_nonempty(i + 1)
        if j < width and PERCENT_CELL_RE.match(row[j].strip()):
            value += "%"
            i = j

        label = ""
        for col in range(span_start, i + 1):
            if col < len(labels) and labels[col]:
                label = labels[col]
                break

        cells.append(f"{label}: {value}" if label else value)
        i += 1

    return " | ".join(escape_markdown_cell(cell) for cell in cells)


def render_table_lines(
    rows: list[list[str]],
    header_rows: int,
) -> tuple[list[str], int, str]:
    """
    Serialise a grid as header-prepended rows.

    Returns (lines, header_line_count, mode) where mode is one of
    "header-prepended", "pipe" (no usable column labels), "pipe-oversized-label"
    (data rows were swallowed into the header block) or "pipe-numeric-guard"
    (re-serialisation would have dropped a figure). The header block is emitted
    as its own line(s) so split_large_table keeps repeating it on every piece.
    """

    labels, banners = _column_labels(rows, header_rows)

    if not any(labels):
        fallback_header = min(max(header_rows, 1), len(rows)) if rows else 0
        return render_table_rows(rows), fallback_header, "pipe"

    if any(len(label) > MAX_COLUMN_LABEL_CHARS for label in labels):
        fallback_header = min(max(header_rows, 1), len(rows)) if rows else 0
        return render_table_rows(rows), fallback_header, "pipe-oversized-label"

    header_lines: list[str] = []
    if banners:
        header_lines.append(escape_markdown_cell(" ".join(banners)))
    header_lines.append(
        "Columns: " + " | ".join(escape_markdown_cell(label) for label in labels if label)
    )

    data_lines: list[str] = []
    for row in rows[header_rows:]:
        line = _render_data_row(row, labels)
        if line:
            data_lines.append(line)

    lines = [*header_lines, *data_lines]

    # Safety net: never let re-serialisation drop a figure. This fires when
    # infer_table_header_rows swallows data rows into the header block and the
    # per-column dedupe then collapses a repeated value. Falling back to the
    # pipe grid keeps the table verbatim rather than silently losing a number.
    if _numeric_tokens(rows_text(rows)) - _numeric_tokens("\n".join(lines)):
        fallback_header = min(max(header_rows, 1), len(rows)) if rows else 0
        return render_table_rows(rows), fallback_header, "pipe-numeric-guard"

    return lines, len(header_lines), "header-prepended"


def infer_table_header_rows(rows: list[list[str]], row_has_th: list[bool]) -> int:
    """
    Estimate how many initial rows should be repeated when an oversized table
    is split. Prefer explicit <th> rows; otherwise infer up to three initial
    non-data rows.
    """

    if not rows:
        return 0

    explicit = 0
    for flag in row_has_th:
        if flag:
            explicit += 1
        else:
            break

    if explicit:
        return min(explicit, 4)

    inferred = 0
    for row in rows[:3]:
        nonempty = [cell for cell in row if cell.strip()]
        if not nonempty:
            inferred += 1
            continue

        numeric = sum(text_looks_numeric(cell) for cell in nonempty)
        numeric_ratio = numeric / len(nonempty)

        # Header/title rows usually contain mostly labels/period names rather
        # than data values. Stop once a row looks data-like.
        if numeric_ratio < 0.40:
            inferred += 1
        else:
            break

    return max(1, inferred)


def is_toc_like_table(rows: list[list[str]]) -> bool:
    """Identify navigation/TOC tables so they do not enter the retrieval index."""

    if not rows:
        return False

    flat_rows = [clean_text(" ".join(row)) for row in rows]
    structural_rows = sum(
        1
        for text in flat_rows
        if re.match(r"^(PART\s+[IVX]+|ITEM\s+\d+[A-Z]?)\b", text, re.IGNORECASE)
    )

    pageish_rows = sum(
        1
        for row in rows
        if row and text_looks_numeric(row[-1])
    )

    return structural_rows >= 3 and pageish_rows >= 2


def table_to_element_payload(table: Tag) -> tuple[str, dict[str, Any]]:
    rows, row_has_th = table_to_grid(table)
    if not rows:
        return "", {}

    if is_toc_like_table(rows):
        return "", {"skipped_toc_table": True}

    caption_tag = table.find("caption")
    caption = clean_text(caption_tag.get_text(" ", strip=True)) if caption_tag else ""

    header_rows = infer_table_header_rows(rows, row_has_th)
    row_lines, header_line_count, serialization = render_table_lines(rows, header_rows)

    text_lines: list[str] = []
    if caption:
        text_lines.append(f"Table: {caption}")
    text_lines.extend(row_lines)

    metadata: dict[str, Any] = {
        "table_row_count": len(rows),
        "table_column_count": max(len(row) for row in rows),
        "table_header_rows": header_line_count,
        "table_header_grid_rows": header_rows,
        "table_serialization": serialization,
        "table_caption": caption or None,
        "table_row_lines": row_lines,
    }
    metadata.update(extract_xbrl_metadata(table))

    return "\n".join(text_lines), metadata


# ============================================================
# SEC HTML -> SEMANTIC ELEMENTS
# ============================================================


def _dedupe_consecutive(elements: list[Element]) -> list[Element]:
    if not elements:
        return []

    out = [elements[0]]
    for element in elements[1:]:
        prev = out[-1]
        same = (
            element.element_type == prev.element_type
            and normalise_key(element.text) == normalise_key(prev.text)
            and element.part == prev.part
            and element.item == prev.item
            and element.section == prev.section
        )
        if not same:
            out.append(element)
    return out


def parse_sec_html(path: str | Path) -> list[Element]:
    """
    Parse an SEC HTML/iXBRL filing into ordered semantic elements.

    Key behaviours:
      - skips hidden/non-content HTML
      - avoids duplicate text from nested container divs
      - suppresses duplicate TOC Part/Item headings by keeping the last
        heading-like occurrence of each structural marker
      - keeps tables as standalone structured elements
      - tracks Part -> Item -> subsection context
      - attaches lightweight iXBRL identifiers where visible
    """

    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    remove_noncontent_html(soup)

    blocks = [tag for tag in soup.find_all(SEMANTIC_TAGS) if should_process_block(tag)]

    # Find canonical (last) structural heading occurrences. This suppresses
    # duplicate Item/Part labels in the table of contents for typical 10-Ks.
    #
    # For Items the last occurrence alone is not enough: filings that print a
    # running page header ("PART II" / "Item 7") on every page repeat the bare
    # Item label long after the section actually starts, which would push the
    # canonical index to the end of the section and leave the whole section
    # tagged with the *previous* Item. Prefer the last titled occurrence, and
    # only fall back to the last bare one when a number never appears titled.
    canonical_part_idx: dict[str, int] = {}
    canonical_item_idx: dict[str, int] = {}
    titled_item_idx: dict[str, int] = {}

    for idx, tag in enumerate(blocks):
        if tag.name == "table":
            text = heading_table_text(tag)
            if not text:
                continue
        else:
            text = clean_text(tag.get_text(" ", strip=True))
            if not text or not is_structural_heading_candidate(tag, text):
                continue

        part = detect_part(text)
        if part:
            canonical_part_idx[part.casefold()] = idx

        item = detect_item(text)
        if item:
            canonical_item_idx[item_number(item)] = idx
            if is_titled_item_heading(text):
                titled_item_idx[item_number(item)] = idx

    canonical_item_idx.update(titled_item_idx)

    elements: list[Element] = []
    current_part: Optional[str] = None
    current_item: Optional[str] = None
    current_section: Optional[str] = None

    for idx, tag in enumerate(blocks):
        anchor = get_anchor(tag)
        is_heading_table = False

        if tag.name == "table":
            heading_text = heading_table_text(tag)
            if heading_text:
                # Structure, not data: fall through to the Part/Item handling
                # below instead of emitting a table element.
                text = heading_text
                is_heading_table = True
            else:
                table_text, metadata = table_to_element_payload(tag)
                if not table_text:
                    continue

                elements.append(
                    Element(
                        element_type="table",
                        text=table_text,
                        part=current_part,
                        item=current_item,
                        section=current_section,
                        source="html",
                        source_anchor=anchor,
                        metadata=metadata,
                    )
                )
                continue
        else:
            text = clean_text(tag.get_text(" ", strip=True))
            if not text:
                continue

        heading_candidate = is_heading_table or is_structural_heading_candidate(tag, text)

        part = detect_part(text) if heading_candidate else None
        if part:
            if canonical_part_idx.get(part.casefold()) != idx:
                continue

            current_part = part
            current_item = None
            current_section = None

            elements.append(
                Element(
                    element_type="heading",
                    text=text,
                    part=current_part,
                    item=None,
                    section=None,
                    source="html",
                    source_anchor=anchor,
                    metadata={"heading_level": "part"},
                )
            )
            continue

        item = detect_item(text) if heading_candidate else None
        if item:
            if canonical_item_idx.get(item_number(item)) != idx:
                continue

            current_item = item
            current_section = None

            elements.append(
                Element(
                    element_type="heading",
                    text=text,
                    part=current_part,
                    item=current_item,
                    section=None,
                    source="html",
                    source_anchor=anchor,
                    metadata={"heading_level": "item"},
                )
            )
            continue

        if normalise_key(text) in {"table of contents", "table of content"}:
            continue

        if is_probable_subsection_heading(tag, text):
            current_section = text
            elements.append(
                Element(
                    element_type="heading",
                    text=text,
                    part=current_part,
                    item=current_item,
                    section=current_section,
                    source="html",
                    source_anchor=anchor,
                    metadata={"heading_level": "section"},
                )
            )
            continue

        element_type = "list" if tag.name == "li" else "paragraph"
        metadata = extract_xbrl_metadata(tag)

        elements.append(
            Element(
                element_type=element_type,
                text=text,
                part=current_part,
                item=current_item,
                section=current_section,
                source="html",
                source_anchor=anchor,
                metadata=metadata,
            )
        )

    return _dedupe_consecutive(elements)


# ============================================================
# TOKEN HELPERS
# ============================================================


def token_ids(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def count_tokens(text: str, tokenizer) -> int:
    return len(token_ids(text, tokenizer))


def decode_tokens(ids: list[int], tokenizer) -> str:
    return clean_text(tokenizer.decode(ids, skip_special_tokens=True))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


# ============================================================
# PREFIX / CONTEXT
# ============================================================


def build_prefix(
    doc_label: str,
    fiscal_period: str,
    part: Optional[str],
    item: Optional[str],
    section: Optional[str],
) -> str:
    path = build_section_path(part, item, section)
    prefix = f"{doc_label} | {fiscal_period}"
    if path:
        prefix += " | " + " > ".join(path)
    return prefix


# ============================================================
# OVERSIZED NARRATIVE SPLITTING
# ============================================================


def _split_by_token_budget(
    text: str,
    prefix: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    """Final fallback that guarantees each piece fits the token budget."""

    prefix_tokens = count_tokens(prefix + "\n\n", tokenizer)
    budget = max_tokens - prefix_tokens
    if budget <= 0:
        raise ValueError(
            f"Prefix alone uses {prefix_tokens} tokens, exceeding max_tokens={max_tokens}."
        )

    ids = token_ids(text, tokenizer)
    pieces = []
    for start in range(0, len(ids), budget):
        piece = decode_tokens(ids[start : start + budget], tokenizer)
        if piece:
            pieces.append(piece)
    return pieces


def split_large_text(
    text: str,
    prefix: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    """
    Prefer sentence boundaries; then word boundaries; finally token slices.
    Every returned piece is guaranteed to fit with the prefix.
    """

    full = f"{prefix}\n\n{text}"
    if count_tokens(full, tokenizer) <= max_tokens:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return _split_by_token_budget(text, prefix, tokenizer, max_tokens)

    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if count_tokens(f"{prefix}\n\n{candidate}", tokenizer) <= max_tokens:
            current.append(sentence)
            continue

        if current:
            pieces.append(" ".join(current))
            current = []

        if count_tokens(f"{prefix}\n\n{sentence}", tokenizer) <= max_tokens:
            current = [sentence]
            continue

        # Oversized single sentence: try word packing.
        words = sentence.split()
        word_piece: list[str] = []

        for word in words:
            candidate = " ".join(word_piece + [word])
            if count_tokens(f"{prefix}\n\n{candidate}", tokenizer) <= max_tokens:
                word_piece.append(word)
            else:
                if word_piece:
                    pieces.append(" ".join(word_piece))
                    word_piece = []

                if count_tokens(f"{prefix}\n\n{word}", tokenizer) <= max_tokens:
                    word_piece = [word]
                else:
                    pieces.extend(
                        _split_by_token_budget(word, prefix, tokenizer, max_tokens)
                    )

        if word_piece:
            pieces.append(" ".join(word_piece))

    if current:
        pieces.append(" ".join(current))

    final: list[str] = []
    for piece in pieces:
        if count_tokens(f"{prefix}\n\n{piece}", tokenizer) <= max_tokens:
            final.append(piece)
        else:
            final.extend(_split_by_token_budget(piece, prefix, tokenizer, max_tokens))

    return final


# ============================================================
# OVERSIZED TABLE SPLITTING
# ============================================================


def _table_preamble_and_rows(element: Element) -> tuple[list[str], list[str], int]:
    lines = [line for line in element.text.splitlines() if line.strip()]
    caption = element.metadata.get("table_caption")

    preamble: list[str] = []
    row_lines = list(element.metadata.get("table_row_lines") or [])

    if caption:
        preamble.append(f"Table: {caption}")

    if not row_lines:
        row_lines = [line for line in lines if line.startswith("|")]

    header_count = int(element.metadata.get("table_header_rows") or 1)
    header_count = min(max(header_count, 1), len(row_lines)) if row_lines else 0

    return preamble, row_lines, header_count


def split_large_table(
    element: Element,
    prefix: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    """
    Keep a table whole when possible. If oversized, split by logical row
    groups and repeat caption/header rows in every child chunk.
    """

    if count_tokens(f"{prefix}\n\n{element.text}", tokenizer) <= max_tokens:
        return [element.text]

    preamble, row_lines, header_count = _table_preamble_and_rows(element)
    if not row_lines:
        return split_large_text(element.text, prefix, tokenizer, max_tokens)

    header_lines = row_lines[:header_count]
    data_lines = row_lines[header_count:]

    repeated = preamble + header_lines
    repeated_text = "\n".join(repeated)

    if count_tokens(f"{prefix}\n\n{repeated_text}", tokenizer) >= max_tokens:
        # Pathological header: preserve correctness by falling back to token
        # slices rather than silently exceeding the embedding limit.
        return _split_by_token_budget(element.text, prefix, tokenizer, max_tokens)

    pieces: list[str] = []
    current_rows: list[str] = []

    # Header-only table.
    if not data_lines:
        return [repeated_text]

    for row in data_lines:
        candidate_lines = repeated + current_rows + [row]
        candidate = "\n".join(candidate_lines)

        if count_tokens(f"{prefix}\n\n{candidate}", tokenizer) <= max_tokens:
            current_rows.append(row)
            continue

        if current_rows:
            pieces.append("\n".join(repeated + current_rows))
            current_rows = []

        single_row = "\n".join(repeated + [row])
        if count_tokens(f"{prefix}\n\n{single_row}", tokenizer) <= max_tokens:
            current_rows = [row]
        else:
            # One physical row is itself too large. Preserve the repeated
            # table context and split only the row text.
            row_prefix = f"{prefix}\n\n{repeated_text}"
            row_pieces = _split_by_token_budget(row, row_prefix, tokenizer, max_tokens)
            for row_piece in row_pieces:
                pieces.append("\n".join(repeated + [row_piece]))

    if current_rows:
        pieces.append("\n".join(repeated + current_rows))

    return pieces


# ============================================================
# FINAL STRUCTURE-AWARE CHUNKER
# ============================================================


def merge_xbrl_metadata(elements: Iterable[Element]) -> dict[str, Any]:
    """Union the lightweight iXBRL identifiers carried by a group of elements."""

    merged: dict[str, Any] = {}
    concepts: set[str] = set()
    contexts: set[str] = set()
    units: set[str] = set()
    fact_count = 0

    for e in elements:
        concepts.update(e.metadata.get("xbrl_concepts", []))
        contexts.update(e.metadata.get("xbrl_context_refs", []))
        units.update(e.metadata.get("xbrl_units", []))
        fact_count += int(e.metadata.get("xbrl_fact_count", 0) or 0)

    if concepts:
        merged["xbrl_concepts"] = sorted(concepts)
    if contexts:
        merged["xbrl_context_refs"] = sorted(contexts)
    if units:
        merged["xbrl_units"] = sorted(units)
    if fact_count:
        merged["xbrl_fact_count"] = fact_count

    return merged


def build_chunks(
    elements: list[Element],
    tokenizer,
    *,
    doc_label: str,
    company: str,
    filing_type: str,
    fiscal_period: str,
    filing_key: Optional[str] = None,
    filing_date: Optional[str] = None,
    cik: Optional[str] = None,
    accession: Optional[str] = None,
    source_url: Optional[str] = None,
    max_tokens: int = 430,
    overlap_paragraphs: int = 0,
    max_overlap_tokens: int = 120,
) -> list[dict[str, Any]]:
    """
    Convert semantic elements into token-bounded chunks.

    Rules:
      1. Never cross Part/Item/subsection boundaries.
      2. Keep tables standalone.
      3. Pack whole adjacent paragraphs/lists when they fit.
      4. Split oversized narrative at sentence/word/token boundaries.
      5. Split oversized tables by row groups and repeat headers.
      6. Prepend deterministic filing/section context to every chunk.
      7. Optional overlap reuses whole trailing paragraphs only; it never
         crosses a structural boundary.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs cannot be negative")

    # chunk_id = {filing_key}:{element_index}:{piece_no}. Keyed on the source
    # element's position so the id survives re-serialisation of the text.
    filing_id = filing_key or re.sub(r"[^A-Za-z0-9]+", "_", doc_label).strip("_")
    element_positions = {id(element): i for i, element in enumerate(elements)}

    def chunk_id_for(element: Element, piece_no: int = 1) -> str:
        return f"{filing_id}:{element_positions.get(id(element), -1)}:{piece_no}"

    chunks: list[Chunk] = []

    current_elements: list[Element] = []
    current_key: tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)

    def make_chunk(
        body: str,
        *,
        chunk_id: str,
        part: Optional[str],
        item: Optional[str],
        section: Optional[str],
        element_type: str,
        source: str,
        source_anchor: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        prefix = build_prefix(doc_label, fiscal_period, part, item, section)
        full_text = f"{prefix}\n\n{body}".strip()
        n_tokens = count_tokens(full_text, tokenizer)

        if n_tokens > max_tokens:
            raise ValueError(
                f"Chunk exceeded max_tokens: {n_tokens} > {max_tokens}. "
                f"Section={section!r}, type={element_type!r}"
            )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=full_text,
                doc=doc_label,
                company=company,
                filing_type=filing_type,
                fiscal_period=fiscal_period,
                filing_date=filing_date,
                cik=cik,
                accession=accession,
                source_url=source_url,
                part=part,
                item=item,
                section=section,
                section_path=build_section_path(part, item, section),
                element_type=element_type,
                source=source,
                source_anchor=source_anchor,
                token_count=n_tokens,
                metadata=metadata or {},
            )
        )

    def flush(allow_overlap: bool = True) -> None:
        nonlocal current_elements

        if not current_elements:
            return

        part, item, section = current_key
        body = "\n\n".join(e.text for e in current_elements)
        source_anchor = next((e.source_anchor for e in current_elements if e.source_anchor), None)

        merged_meta = merge_xbrl_metadata(current_elements)

        make_chunk(
            body,
            chunk_id=chunk_id_for(current_elements[0]),
            part=part,
            item=item,
            section=section,
            element_type="narrative",
            source="html",
            source_anchor=source_anchor,
            metadata=merged_meta,
        )

        overlap: list[Element] = []
        if allow_overlap and overlap_paragraphs:
            for e in reversed(current_elements):
                if len(overlap) >= overlap_paragraphs:
                    break
                prefix = build_prefix(doc_label, fiscal_period, e.part, e.item, e.section)
                if count_tokens(f"{prefix}\n\n{e.text}", tokenizer) <= max_overlap_tokens:
                    overlap.append(e)
                else:
                    break
            overlap.reverse()

        current_elements = overlap

    for element in elements:
        if element.element_type == "heading":
            flush(allow_overlap=False)
            current_key = (element.part, element.item, element.section)
            continue

        key = (element.part, element.item, element.section)

        if key != current_key:
            flush(allow_overlap=False)
            current_key = key

        prefix = build_prefix(
            doc_label,
            fiscal_period,
            element.part,
            element.item,
            element.section,
        )

        if element.element_type == "table":
            table_pieces = split_large_table(
                element,
                prefix,
                tokenizer,
                max_tokens,
            )

            # A narrative lead-in ("The following table shows net sales by
            # category...") introduces the table that follows it. Flushing it
            # on its own produces a tiny, number-free chunk that still matches
            # the natural query and outranks the table holding the answer, so
            # fold it into the first table piece when it fits.
            caption_elements: list[Element] = []
            if current_elements and current_key == key and table_pieces:
                caption_body = "\n\n".join(e.text for e in current_elements)
                merged_first = f"{caption_body}\n\n{table_pieces[0]}"

                if count_tokens(f"{prefix}\n\n{merged_first}", tokenizer) <= max_tokens:
                    caption_elements = current_elements
                    table_pieces = [merged_first, *table_pieces[1:]]
                    current_elements = []

            # No-op when the caption was folded in above.
            flush(allow_overlap=False)

            caption_meta = merge_xbrl_metadata(caption_elements)
            caption_anchor = next(
                (e.source_anchor for e in caption_elements if e.source_anchor), None
            )

            for piece_no, piece in enumerate(table_pieces, start=1):
                meta = dict(element.metadata)
                meta.pop("table_row_lines", None)
                meta["table_piece"] = piece_no
                meta["table_piece_count"] = len(table_pieces)

                if caption_elements and piece_no == 1:
                    meta["caption_folded"] = True
                    for field_name, value in caption_meta.items():
                        if field_name == "xbrl_fact_count":
                            meta[field_name] = int(meta.get(field_name, 0) or 0) + value
                        else:
                            meta[field_name] = sorted(
                                set(meta.get(field_name, [])) | set(value)
                            )

                make_chunk(
                    piece,
                    chunk_id=chunk_id_for(element, piece_no),
                    part=element.part,
                    item=element.item,
                    section=element.section,
                    element_type="table",
                    source=element.source,
                    source_anchor=element.source_anchor or caption_anchor,
                    metadata=meta,
                )
            continue

        # Narrative/list element: first try adding the whole element to the
        # current chunk.
        candidate_elements = current_elements + [element]
        candidate_body = "\n\n".join(e.text for e in candidate_elements)

        if count_tokens(f"{prefix}\n\n{candidate_body}", tokenizer) <= max_tokens:
            current_elements.append(element)
            continue

        flush(allow_overlap=True)

        # An overlap paragraph may have been carried forward. Try again with
        # the new element. If that still does not fit, drop the overlap before
        # splitting the new element.
        if current_elements:
            candidate_body = "\n\n".join(
                [*(e.text for e in current_elements), element.text]
            )
            if count_tokens(f"{prefix}\n\n{candidate_body}", tokenizer) <= max_tokens:
                current_elements.append(element)
                continue
            current_elements = []

        if count_tokens(f"{prefix}\n\n{element.text}", tokenizer) <= max_tokens:
            current_elements = [element]
            continue

        # Oversized single paragraph/list: split it into bounded child chunks.
        pieces = split_large_text(
            element.text,
            prefix,
            tokenizer,
            max_tokens,
        )

        for piece_no, piece in enumerate(pieces, start=1):
            meta = dict(element.metadata)
            meta["narrative_piece"] = piece_no
            meta["narrative_piece_count"] = len(pieces)

            make_chunk(
                piece,
                chunk_id=chunk_id_for(element, piece_no),
                part=element.part,
                item=element.item,
                section=element.section,
                element_type=element.element_type,
                source=element.source,
                source_anchor=element.source_anchor,
                metadata=meta,
            )

    flush(allow_overlap=False)

    return [asdict(chunk) for chunk in chunks]


# ============================================================
# VALIDATION / INSPECTION HELPERS
# ============================================================


def validate_chunks(chunks: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    oversized = [
        (i, chunk["token_count"])
        for i, chunk in enumerate(chunks)
        if chunk["token_count"] > max_tokens
    ]

    missing_section_path = [
        i for i, chunk in enumerate(chunks) if "section_path" not in chunk
    ]

    cross_item = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        item_markers = set(
            m.group(1).upper()
            for m in re.finditer(r"\bItem\s+(\d+[A-Z]?)\b", text, re.IGNORECASE)
        )
        # Not every mention of another Item implies a bad chunk, so this is a
        # warning only, not a hard validation failure.
        declared = chunk.get("item")
        declared_number = item_number(declared) if declared else None
        unexpected = {x for x in item_markers if x != declared_number}
        if unexpected:
            cross_item.append((i, sorted(unexpected)))

    return {
        "chunk_count": len(chunks),
        "oversized": oversized,
        "missing_section_path": missing_section_path,
        "cross_item_mentions_warning": cross_item,
        "max_token_count": max((c["token_count"] for c in chunks), default=0),
    }


def element_type_counts(elements: Iterable[Element]) -> dict[str, int]:
    return dict(Counter(e.element_type for e in elements))


def chunk_type_counts(chunks: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(c["element_type"] for c in chunks))


# ============================================================
# PDF INSPECTION HELPERS (FALLBACK / DIAGNOSTICS)
# ============================================================


def parse_pdf_blocks(path: str | Path) -> list[dict[str, Any]]:
    """Extract positioned PDF text blocks for Stage 3 parser inspection."""

    import pymupdf

    doc = pymupdf.open(path)
    blocks: list[dict[str, Any]] = []

    for pno, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text = clean_text(text)
            if text:
                blocks.append(
                    {
                        "page": pno,
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "text": text,
                    }
                )

    return blocks


def find_pdf_boilerplate(
    blocks: list[dict[str, Any]],
    threshold: float = 0.5,
) -> set[str]:
    """
    Find exact block texts repeated on more than `threshold` of PDF pages.
    This is intentionally simple and used for inspection, not HTML parsing.
    """

    pages = {b["page"] for b in blocks}
    if not pages:
        return set()

    pages_per_text: defaultdict[str, set[int]] = defaultdict(set)
    for block in blocks:
        pages_per_text[block["text"]].add(block["page"])

    cutoff = len(pages) * threshold
    return {
        text
        for text, seen_pages in pages_per_text.items()
        if len(seen_pages) > cutoff
    }


def find_sparse_pdf_pages(
    path: str | Path,
    min_chars: int = 100,
) -> list[int]:
    """Return 1-indexed PDF pages with very little extractable text."""

    import pymupdf

    doc = pymupdf.open(path)
    return [
        pno + 1
        for pno, page in enumerate(doc)
        if len(page.get_text().strip()) < min_chars
    ]
