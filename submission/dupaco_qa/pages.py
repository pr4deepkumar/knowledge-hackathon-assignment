# Turns raw pages into clean text, sections and tables, and searches the sections by keyword.
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional
from urllib.parse import urlparse

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
MONTH_DATE = re.compile(r"\b(" + "|".join(MONTHS) + r")\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Mapping typographic dashes and quotes, the non-breaking hyphen and the no-break space to plain ASCII
_PUNCT = str.maketrans({"–": "-", "—": "-", "\u2011": "-", "’": "'", "‘": "'",
                        "“": '"', "”": '"', "\xa0": " "})


def clean(s: str):
    # Turning text plain: ASCII dashes and quotes, no markdown escapes, bold markers or literal "\r\n", single spaces
    s = (s or "").translate(_PUNCT)
    s = s.replace("\\*", "*").replace("\\#", "#").replace("\\_", "_").replace("\\\\r\\\\n", " ")
    s = s.replace("\\r\\n", " ").replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", s).strip()


def norm(s: str):
    # Making the lower-case form used for matching: cleaned, no asterisks or thousands separators, unspaced dashes
    s = clean(s).lower().replace("*", "")
    s = re.sub(r"(\d),(?=\d{3})", r"\1", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s.strip()


def strip_links(s: str):
    # Replacing markdown links with their text and dropping images
    return MD_LINK.sub(lambda m: "" if m.group(0).startswith("!") else m.group(1), s)


def sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text) if s.strip()]


def month_date(s: str):
    m = MONTH_DATE.search(s or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), MONTHS.index(m.group(1).lower()) + 1, int(m.group(2)))
    except ValueError:
        return None


def slash_date(s: str):
    m = SLASH_DATE.search(s or "")
    if not m:
        return None
    month, day, year = (int(part) for part in m.groups())
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def iso(d: Optional[date]):
    return d.isoformat() if d else None


def money(s: str):
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", s or "")
    return float(m.group(1).replace(",", "")) if m else None


def fmt_money(x: Optional[float]):
    if x is None:
        return ""
    return f"${int(x):,}" if float(x).is_integer() else f"${x:,.2f}"


# Ranking how far each page type is trusted to state a current rate; the higher rank wins
TYPE_RANK = {"rates": 5, "product": 4, "promo": 2, "guide": 1, "calculator": 1, "news": 1,
             "doc": 1, "blog": 0, "legal": 0, "listing": 0}

_PROMO_PAGES = {"/switch-and-earn", "/offer-expired", "/earn-more", "/checking", "/mortgage-refi",
                "/money-makeover", "/21-day-sprint-to-savings-challenge", "/sprint-to-savings-challenge",
                "/thankuse", "/kccc", "/kccc-membership-grant"}
_GUIDE_PAGES = {"/homebuying-guidance"}
_LISTING_PAGES = {"/resources/personal-resources", "/learn/blog", "/newsroom", "/business/resources",
                  "/calculators", "/business-calculators"}


def url_key(url: str):
    # Making a page's key: its lower-case path, without query, fragment or trailing slash
    p = urlparse(url if "//" in url else "https://" + url)
    return p.path.rstrip("/").lower() or "/"


def page_type(key: str):
    if key in ("/rates", "/business-rates"):
        return "rates"
    if key in _LISTING_PAGES or re.search(r"/(tag|author|category)(/|$)|/page/\d+$", key):
        return "listing"
    if key.startswith("/hubfs/"):
        return "doc"
    if key.startswith(("/calculators", "/business-calculators")):
        return "calculator"
    if key.startswith("/newsroom/"):
        return "news"
    if key.startswith(("/learn/blog/", "/business/resources/", "/dupacodaily/")) or re.match(r"^/\d{4}/\d{2}/", key):
        return "blog"
    if key.endswith("-lp") or key in _PROMO_PAGES or re.search(r"(^|/|-)(promos?|promotions?|offers?)(-|/|$)", key):
        return "promo"
    if key in _GUIDE_PAGES or key.startswith("/learn/master-your-money"):
        return "guide"
    if key.startswith(("/privacy-policy", "/disclosures", "/about/disclosures", "/digital-privacy-policy",
                       "/photo-release-form", "/subscription-preferences")):
        return "legal"
    return "product"


def segment(key: str):
    return "business" if key.startswith("/business") else "personal"


def publish_date(key: str, ptype: str, markdown: str):
    # Finding a news or blog post's date: the first "Month D, YYYY" near the top, else the date in its URL
    if ptype not in ("news", "blog"):
        return None
    found = month_date("\n".join(markdown.splitlines()[:40]))
    if found:
        return found
    m = re.match(r"^/(\d{4})/(\d{2})/(\d{2})/", key)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# Matching lines that are never content on any page, web-form field labels and pickers included
_JUNK = re.compile(
    r"^(✕|load more|sort by:.*|#+ filter|third party plugins|messenger launcher|\* \* \*|#+|\\# # #|"
    r"\d+ min read|keep reading|read more|skip to main content)$|^!\[[^\]]*\]\([^)]*\)$|^\[!\[|"
    r"^[\w ,.'/&()?-]{1,60}\\?\*$|^please select|^(drop us|share) your info below|^submit$",
    re.I,
)
# Matching headings over related-article cards, which run until the next heading at the same or a higher level
_RELATED = re.compile(r"^#{1,4}\s*(handpicked for you|recent news|related articles|you might also like)\s*$", re.I)


def boilerplate_lines(markdowns: list[str], min_share: float = 0.04, max_len: int = 160):
    counts = Counter()
    for md in markdowns:
        counts.update({line.strip() for line in md.splitlines() if line.strip()})
    floor = max(25, int(min_share * max(1, len(markdowns))))
    return {line for line, n in counts.items() if n >= floor and len(line) <= max_len}


@dataclass
class Section:
    heading: str        # such as "Page title > H2 > H3"
    text: str           # cleaned, with links reduced to their text


@dataclass
class PageVersion:
    key: str
    url: str
    title: str
    snap: int
    ptype: str
    segment: str
    published: Optional[date]
    markdown: str       # the raw page, for the table and fact parsers
    sections: list[Section] = field(default_factory=list)


def clean_sections(title: str, markdown: str, boiler: set[str]):
    """The page split at its headings into searchable sections, without boilerplate, junk lines or lists of
    related articles."""
    sections = []
    headings = []       # Tracking the open headings, as (level, text)
    body = []           # Collecting the current section's lines
    skip_level = 0      # Remembering the level of a related-articles heading while its cards are skipped

    def flush():
        text = "\n".join(body).strip()
        if len(text) >= 20:
            sections.append(Section(" > ".join([clean(title)] + [name for _, name in headings]), text))
        body.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = HEADING.match(line)
        if _RELATED.match(line):
            # Skipping only the article cards, since FAQs often follow them
            flush()
            skip_level = len(line) - len(line.lstrip("#"))
            continue
        if skip_level:
            # Skipping the cards until a heading at the related list's level or higher
            if not heading or len(heading.group(1)) > skip_level:
                continue
            skip_level = 0
        if line in boiler or _JUNK.match(line):
            continue
        if heading:
            flush()
            _push_heading(headings, heading)
            continue
        text = clean(strip_links(line))
        if text:
            body.append(text)
        if len(body) >= 40:
            # Splitting long sections so they stay a readable size for search and quoting
            flush()
    flush()
    return sections


_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP = re.compile(r"^\s*\|\s*:?-{3,}")
# Matching "Rates are effective as of 08/31/2026", but not a footnote such as "1Rates are effective as of ..."
_EFFECTIVE = re.compile(r"^\W*rates?\s+(?:are\s+)?effective\s+(?:as\s+of\s+)?(\d{1,2}/\d{1,2}/\d{2,4})", re.I)


@dataclass
class Table:
    title: str                  # the label just above the table, such as "Share Savings"
    headings: list[str]         # the markdown headings the table is under
    header: list[str]
    rows: list[list[str]]
    effective: Optional[date]   # from "Rates are effective as of ..." below the table
    line: int
    footer: str = ""            # the first lines below the table; footnotes often name the product


def find_tables(markdown: str):
    lines = markdown.splitlines()
    tables = []
    headings = []       # Tracking the open headings, as (level, text)
    i = 0
    while i < len(lines):
        heading = HEADING.match(lines[i].strip())
        if heading:
            _push_heading(headings, heading)
        if not (_ROW.match(lines[i]) and i + 1 < len(lines) and _SEP.match(lines[i + 1])):
            i += 1
            continue
        header = _cells(lines[i])
        end = i + 2
        rows = []
        while end < len(lines) and _ROW.match(lines[end]):
            rows.append(_cells(lines[end]))
            end += 1
        title = _title_above(lines, i)
        effective, footer = _below_table(lines, end)
        tables.append(Table(title, [name for _, name in headings], header, rows, effective, i, footer))
        i = end
    return tables


def _push_heading(headings: list, heading: re.Match):
    # Closing the open headings at this level or deeper, then opening this one
    level = len(heading.group(1))
    while headings and headings[-1][0] >= level:
        headings.pop()
    headings.append((level, clean(strip_links(heading.group(2)))))


def _cells(line: str):
    return [clean(strip_links(cell)) for cell in line.strip().strip("|").split("|")]


def _title_above(lines: list, start: int):
    """The nearest label-like line in the 39 lines above a table; anything else, including the rows of layout
    tables, is stepped over. The previous rate table's "effective as of" footer ends the search."""
    for k in range(start - 1, max(-1, start - 40), -1):
        line = lines[k].strip()
        if _ROW.match(line) or not line:
            continue
        if _EFFECTIVE.match(line):
            return ""
        if _is_title(line):
            return clean(strip_links(line)).strip("#").strip()
    return ""


def _is_title(line: str):
    # Checking for a short label: not a bullet, sentence, footnote, table row or "effective as of" line
    s = clean(strip_links(line)).strip("#").strip()
    if not s or len(s) > 70 or _ROW.match(line) or s.startswith(("-", "*", "+")):
        return False
    # Rejecting footnotes, which start with their number, but not labels such as "1st Rate Checking"
    if s[0].isdigit() and not s.lower().startswith(("1st ", "10 ", "17 ")):
        return False
    if s.endswith((".", ":", "!", "?")) or "effective as of" in s.lower():
        return False
    return True


def _below_table(lines: list, end: int):
    effective = None
    footer = []
    for k in range(end, min(len(lines), end + 25)):
        if _ROW.match(lines[k]) and k > end:
            break
        line = lines[k].strip()
        if line and len(footer) < 6:
            footer.append(clean(strip_links(line))[:200])
        m = _EFFECTIVE.match(line)
        if m and effective is None:
            effective = slash_date(m.group(1))
    return effective, " ".join(footer)


# Weighting search results by page type (0 leaves a type out), and counting each page-title word three times
SEARCH_WEIGHT = {"product": 1.0, "rates": 1.0, "news": 0.9, "blog": 0.75, "promo": 0.7, "guide": 0.7,
                 "calculator": 0.6, "doc": 0.6, "legal": 0.3, "listing": 0.0}
TITLE_WEIGHT = 3

_STOP = set("""a an and are as at be by can do does for from has have how i if in is it its me my of on or our
so that the their them there these this to us was we what when where which who why will with you your""".split())
_SUFFIXES = ("izations", "ization", "ations", "ation", "itions", "ition", "ings", "ing", "ions", "ion",
             "ments", "ment", "ized", "izes", "ize", "ied", "ies", "ed", "es", "s", "e")


def stem(word: str):
    # Stripping suffixes crudely, so that "recognized" meets "recognition" and "loan" meets "loans"
    if len(word) <= 4 or not word.isalpha() or word.endswith("ss"):
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokens(text: str):
    # Splitting text into the words search compares: lower-case, stemmed, without stop words
    words = []
    for word in re.findall(r"[a-z0-9$]+(?:[.'][a-z0-9]+)*%?", (text or "").lower()):
        word = word.replace("'", "")
        if word not in _STOP:
            words.append(stem(word))
    return words


@dataclass
class SectionDoc:
    page: str
    url: str
    title: str
    heading: str
    text: str
    ptype: str
    published: Optional[date]
    snaps: set = field(default_factory=set)


class BM25Index:
    """Keyword search ranked by BM25, the standard formula: a query word scores more the rarer it is across
    sections, with diminishing returns for repeats (k1) and less credit in long sections (b)."""
    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[SectionDoc] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._seen: dict[tuple, int] = {}
        self._avg_len = 0.0

    def add(self, doc: SectionDoc, snap: int):
        # Indexing a section once; a later snapshot with the same text only adds its snapshot number
        sig = (doc.page, doc.heading, hash(doc.text))
        if sig in self._seen:
            self.docs[self._seen[sig]].snaps.add(snap)
            return
        doc.snaps.add(snap)
        i = len(self.docs)
        self._seen[sig] = i
        self.docs.append(doc)
        # Keeping the part of the title before the bar, as in "Forbes ranks Dupaco ... | Dupaco"
        title = doc.title.split("|")[0]
        tf = Counter(tokens(title) * TITLE_WEIGHT + tokens(doc.heading) + tokens(doc.text))
        self._tf.append(tf)
        self._len.append(sum(tf.values()))
        for word in tf:
            self._postings[word].append(i)
        # Clearing the cached average section length, which search() recomputes
        self._avg_len = 0.0

    def search(self, query: str, k: int = 10, allow: Optional[Callable[[SectionDoc], bool]] = None,
               boost: Optional[Callable[[SectionDoc], float]] = None):
        # Returning the k best (score, doc) pairs among the docs allow() accepts, each score times boost()
        if not self.docs:
            return []
        if not self._avg_len:
            self._avg_len = sum(self._len) / len(self._len)
        n = len(self.docs)
        scores: dict[int, float] = defaultdict(float)
        # Sorting the words, since scores are float sums and a fixed order breaks ties the same way every run
        for word in sorted(set(tokens(query))):
            postings = self._postings.get(word)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for i in postings:
                f = self._tf[i][word]
                length_norm = 1 - self.b + self.b * self._len[i] / self._avg_len
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * length_norm)
        ranked = []
        for i, score in scores.items():
            doc = self.docs[i]
            if allow and not allow(doc):
                continue
            ranked.append((score * (boost(doc) if boost else 1.0), i))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(score, self.docs[i]) for score, i in ranked[:k]]


def focus(text: str, query: str, max_chars: int):
    """The part of a long passage that best matches the query, so fine print deep in a section isn't cut off.
    That is the line with the most query words, grown into a window of up to max_chars. If the line with the
    rarest query words falls outside that window, it is half a window around each of the two lines instead."""
    if len(text) <= max_chars:
        return text
    words = set(tokens(query))
    lines = text.split("\n")
    line_words = [words & set(tokens(line)) for line in lines]
    # Counting in how many lines each query word appears
    df = Counter(word for found in line_words for word in found)
    most = max(range(len(lines)), key=lambda i: (len(line_words[i]), -i))
    rarest = max(range(len(lines)), key=lambda i: (sum(1 / df[w] for w in sorted(line_words[i])), -i))
    lo, hi = _grow(lines, most, max_chars)
    if lo <= rarest <= hi or len(lines[most]) >= max_chars:
        return _window(lines, most, query, max_chars)
    return "\n".join(_window(lines, i, query, max_chars // 2) for i in sorted((most, rarest)))


def _grow(lines: list, best: int, budget: int):
    lo = hi = best
    size = len(lines[best])
    while True:
        grew = False
        if lo > 0 and size + len(lines[lo - 1]) + 1 <= budget:
            lo -= 1
            size += len(lines[lo]) + 1
            grew = True
        if hi < len(lines) - 1 and size + len(lines[hi + 1]) + 1 <= budget:
            hi += 1
            size += len(lines[hi]) + 1
            grew = True
        if not grew:
            return lo, hi


def _window(lines: list, best: int, query: str, budget: int):
    if len(lines[best]) >= budget:
        # Starting a long line's window so that its first query word sits a third of the way in
        lowered = lines[best].lower()
        query_words = re.findall(r"[a-z0-9$]{4,}", (query or "").lower())
        hits = [pos for pos in (lowered.find(w) for w in query_words) if pos >= 0]
        start = max(0, min(hits) - budget // 3) if hits else 0
        return ("… " if start else "") + lines[best][start:start + budget] + " …"
    lo, hi = _grow(lines, best, budget)
    return ("… " if lo > 0 else "") + "\n".join(lines[lo:hi + 1]) + (" …" if hi < len(lines) - 1 else "")
