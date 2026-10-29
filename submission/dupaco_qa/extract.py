# Pulls facts out of pages: rates, phone and routing numbers, branches, hours, events, claims, figures and links.
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .pages import HEADING, PCT, Table, clean, fmt_money, month_date, norm, sentences, strip_links

"""The site's product names, as (pattern, product, family, specific). Order matters: each text takes the first
pattern it matches, so narrower names come first ("business platinum" before "platinum"). match_product prefers
a specific match from any of its texts over a generic one such as "special" or "regular"."""
_PRODUCTS = [
    (r"share savings", "Share Savings", "savings", True),
    (r"bright savings", "Bright Savings", "savings", True),
    (r"you name it", "You Name It Savings", "savings", True),
    (r"holiday club", "Holiday Club Savings", "savings", True),
    (r"launch ?pad", "Launchpad Savings", "savings", True),
    (r"high-? ?yield savings", "High-Yield Savings", "savings", True),
    (r"health savings|\bhsa\b", "Health Savings Account", "savings", True),
    (r"investor'?s choice", "Investor's Choice Money Market", "money_market", True),
    (r"\bachieve\b", "Achieve Money Market", "money_market", True),
    (r"1st rate checking|first rate checking", "1st Rate Checking", "checking", True),
    (r"free checking", "Free Checking", "checking", True),
    (r"\b(roth|traditional)\b.*\bspecial\b|special iras?", "IRA Certificate Special", "ira", True),
    (r"variable iras?|\b(roth|traditional): ?\$", "Variable IRA", "ira", True),
    (r"^(roth|traditional)\b|regular iras?|\biras?\b", "IRA Certificate", "ira", True),
    (r"certificate special|special certificates?", "Certificate Special", "certificate", True),
    (r"\bcertificates?\b", "Certificate", "certificate", True),
    (r"\bspecial\b", "Certificate Special", "certificate", False),
    (r"\bregular\b", "Certificate", "certificate", False),
    (r"\bauto\b|motorcycle|vehicle", "Auto & Motorcycle Loan", "loan", True),
    (r"\bboat\b|\brv\b", "Boat & RV Loan", "loan", True),
    (r"\batv\b|snowmobile|jet ski", "ATV, Snowmobile & Jet Ski Loan", "loan", True),
    (r"home improvement", "Home Improvement Loan", "loan", True),
    (r"\bheloc\b|home equity line", "Home Equity Line of Credit (HELOC)", "loan", True),
    (r"cash-? ?out|home equity refinance", "Home Equity Refinance", "mortgage", True),
    (r"home equity loan|^loan-standard", "Home Equity Loan", "loan", True),
    (r"rate relief", "Mortgage Rate Relief Program", "mortgage", True),
    (r"home loans?|mortgage", "Home Loan (Mortgage)", "mortgage", True),
    (r"purchase:|refinance:|fixed conventional", "Home Loan (Mortgage)", "mortgage", False),
    (r"personal loans?", "Personal Loan", "loan", True),
    (r"business platinum", "Business Platinum Visa", "card", True),
    (r"business rewards", "Business Rewards Visa", "card", True),
    (r"platinum", "Platinum Visa", "card", True),
    (r"rewards", "Rewards Visa", "card", True),
]
_PRODUCT_RULES = [(re.compile(pattern), name, family, specific) for pattern, name, family, specific in _PRODUCTS]

DEPOSIT_FAMILIES = {"savings", "money_market", "checking", "certificate", "ira", "deposit"}
LENDING_FAMILIES = {"loan", "mortgage", "card"}
VALUE_ROLES = {"dividend_rate", "apy", "apr", "rate", "intro_dividend_rate", "intro_apy"}
METRIC_NAMES = {"dividend_rate": "dividend rate", "apy": "APY", "apr": "APR", "rate": "interest rate",
                "intro_dividend_rate": "introductory dividend rate (first 6 months)",
                "intro_apy": "introductory APY (first 6 months)"}


@dataclass(frozen=True)
class RateFact:
    product: str                 # e.g. "Share Savings", or an unmatched table's own title or heading
    family: str                  # one of DEPOSIT_FAMILIES or LENDING_FAMILIES
    segment: str                 # personal or business
    account: str                 # roth, traditional or ""
    variant: str                 # jumbo, standard, intro or ""
    purpose: str                 # purchase, refinance or ""
    term: Optional[int]          # in months
    tier_kind: str               # balance, loan_amount, model_year or ""
    tier_lo: Optional[float]
    tier_hi: Optional[float]
    metric: str                  # one of VALUE_ROLES
    value: str                   # as shown, e.g. "5.39%" or "12.50%-19.50%"
    nums: tuple
    label: str                   # the row's label cells, or the sentence for a prose fact
    url: str
    page: str
    ptype: str
    snap: int
    effective: Optional[date]
    title: str
    extra: tuple = ()            # other columns as (name, value) pairs, plus prose and expiry markers

    @property
    def key(self):
        return (self.segment, self.product, self.account, self.variant, self.purpose, self.term,
                self.tier_kind, self.tier_lo, self.tier_hi, self.metric)

    @property
    def display_product(self):
        if self.segment == "business" and not self.product.startswith("Business"):
            return "Business " + self.product
        return self.product


def term_text(fact: RateFact):
    if not fact.term:
        return ""
    if fact.family == "mortgage" and fact.term % 12 == 0:
        return f"{fact.term // 12}-year"
    return f"{fact.term}-month"


def tier_text(fact: RateFact):
    # Writing the tier in the site's own wording, such as "2024-27 model years" or "balance $25,000+"
    lo = fact.tier_lo
    hi = fact.tier_hi
    if fact.tier_kind == "model_year":
        if lo is None:
            return f"{int(hi)} model year (and older)"
        if hi is None:
            return f"{int(lo)} model year (and newer)"
        if lo == hi:
            return f"{int(lo)} model year"
        same_century = int(lo) // 100 == int(hi) // 100
        end = f"{int(hi) % 100:02d}" if same_century else f"{int(hi)}"
        return f"{int(lo)}-{end} model years"
    if fact.tier_kind in ("balance", "loan_amount"):
        what = "balance" if fact.tier_kind == "balance" else "loan amount"
        if lo is None:
            return f"{what} up to {fmt_money(hi)}"
        if hi is None:
            return f"{what} {fmt_money(lo)}+"
        return f"{what} {fmt_money(lo)}-{fmt_money(hi)}"
    return ""


def metric_text(fact: RateFact):
    return METRIC_NAMES.get(fact.metric, fact.metric)


def qualifiers(fact: RateFact):
    # Listing what sets this rate apart from the product's other rates, such as "Roth", "jumbo" or "60-month"
    parts = [fact.account.title(), fact.variant, fact.purpose, term_text(fact), tier_text(fact)]
    return [part for part in parts if part]


def column_role(header: str):
    # Stripping footnote markers first, as in "Annual Percentage Yield (APY)1"
    h = re.sub(r"[\d\s*]+$", "", norm(header))
    if "after promotional" in h:
        return "apy" if ("yield" in h or "apy" in h) else "dividend_rate"
    if "1st 6 months" in h:
        return "intro_apy" if ("yield" in h or "apy" in h) else "intro_dividend_rate"
    if "percentage yield" in h or h.startswith("apy"):
        return "apy"
    if "percentage rate" in h or h.startswith("apr") or h == "variable interest rate":
        return "apr"
    if "dividend rate" in h:
        return "dividend_rate"
    if h in ("rate", "interest rate"):
        return "rate"
    if h.startswith(("min", "max")):
        return "extra:" + re.sub(r"[^a-z]+", "_", h.replace(".", "")).strip("_")
    if h == "points" or "payment" in h:
        return "extra:" + ("points" if h == "points" else "payment_example")
    return "label"


def match_product(texts: list[str]):
    # Taking each text's first matching pattern, and returning the first specific result, else the first generic one
    generic = None
    for text in texts:
        text = norm(text)
        for pattern, name, family, specific in _PRODUCT_RULES:
            if pattern.search(text):
                if specific:
                    return name, family, True
                generic = generic or (name, family, False)
                break
    return generic or ("", "other", False)


def table_facts(t: Table, *, url: str, page: str, ptype: str, segment: str, snap: int, page_title: str):
    roles = [column_role(h) for h in t.header]
    if not any(role in VALUE_ROLES for role in roles):
        return []
    header_product = t.header[0] if roles and roles[0] == "label" else ""
    facts = []
    for row in t.rows:
        if len(row) < len(roles):
            row = row + [""] * (len(roles) - len(row))
        labels = [cell for cell, role in zip(row, roles) if role == "label" and cell not in ("", "-", "—")]
        desc = norm(" | ".join(labels))
        product, family = _row_product(desc, header_product, t, page_title, roles)
        fact_segment = "business" if product.startswith("Business") else segment
        account = next((a for a in ("roth", "traditional") if re.match(rf"^{a}\b", desc)), "")
        variant = _row_variant(desc, family)
        purpose = _row_purpose(desc, product)
        tier_kind, lo, hi = _tier(desc, family)
        extra = tuple((role.removeprefix("extra:"), cell) for cell, role in zip(row, roles)
                      if role.startswith("extra:") and cell)
        for cell, role in zip(row, roles):
            if role not in VALUE_ROLES:
                continue
            parsed = _value(cell)
            if not parsed:
                continue
            value, nums = parsed
            if family in LENDING_FAMILIES and role == "apy":
                # Treating an APY column on a loan as APR, since some pages mislabel it
                role = "apr"
            facts.append(RateFact(
                product=product, family=family, segment=fact_segment, account=account, variant=variant,
                purpose=purpose, term=_term(desc), tier_kind=tier_kind, tier_lo=lo, tier_hi=hi, metric=role,
                value=value, nums=nums, label=" | ".join(labels), url=url, page=page, ptype=ptype, snap=snap,
                effective=t.effective, title=t.title, extra=extra))
    return facts


def _row_variant(desc: str, family: str):
    if "intro" in desc:
        return "intro"
    if "jumbo" in desc:
        return "jumbo"
    if family in ("certificate", "ira") and "special" in desc:
        # Calling the non-jumbo special "standard", which is how a question tells it from the jumbo one
        return "standard"
    return ""


def _row_purpose(desc: str, product: str):
    if product == "Home Equity Refinance" or desc.startswith("refinance"):
        return "refinance"
    if desc.startswith("purchase"):
        return "purchase"
    return ""


def _row_product(desc: str, header_product: str, t: Table, page_title: str, roles: list):
    """The product a table row is about: named by the row or anything above it, else by the footnotes below
    or the page title, else the table's title, nearest heading or page title as it stands, with a family
    guessed from the rate columns."""
    product, family, _ = match_product([desc, header_product, t.title, *reversed(t.headings)])
    if not product:
        product, family, _ = match_product([t.footer, page_title])
    if not product:
        product = t.title or (t.headings[-1] if t.headings else page_title)
        family = "deposit" if any(role in ("apy", "dividend_rate", "intro_apy") for role in roles) else "loan"
    return product, family


# Telling which rate a percentage is from the words near it; the earliest entry in this list wins, so APY beats APR
_METRIC_WORDS = [("apy", r"\bapy\b|annual percentage yield"), ("apr", r"\bapr\b|annual percentage rate"),
                 ("dividend_rate", r"dividend rate"), ("rate", r"\binterest rate\b")]
_EXPIRED = re.compile(r"\b(offer|promotion|special)\s+(ended|expired)|\bexpired\b|\bended\b")


def prose_facts(md: str, *, url: str, page: str, ptype: str, segment: str, snap: int, page_title: str):
    # Reading rates stated in sentences or "label: value" lines rather than in tables
    facts = []
    heading = ""
    for raw in md.splitlines():
        line = raw.strip()
        heading_line = HEADING.match(line)
        if heading_line:
            heading = clean(strip_links(heading_line.group(2)))
            continue
        if not line or line.startswith("|") or "%" not in line:
            continue
        for sentence in sentences(clean(strip_links(line))):
            if not PCT.search(sentence) or len(sentence) > 400:
                continue
            product, family, specific = match_product([sentence, heading, page_title])
            if not specific:
                continue
            low = norm(sentence)
            tier_kind, lo, hi = _tier(low, family)
            extra: tuple = (("kind", "prose"),)
            if _EXPIRED.search(low):
                extra += (("expired", "yes"),)
            fact_segment = "business" if product.startswith("Business") else segment
            for m in PCT.finditer(sentence):
                near = norm(sentence[max(0, m.start() - 60):m.end() + 40])
                metric = next((name for name, pattern in _METRIC_WORDS if re.search(pattern, near)), None)
                if not metric:
                    continue
                facts.append(RateFact(
                    product=product, family=family, segment=fact_segment,
                    account="", variant="jumbo" if "jumbo" in low else "", purpose="", term=_term(low),
                    tier_kind=tier_kind, tier_lo=lo, tier_hi=hi, metric=metric, value=f"{m.group(1)}%",
                    nums=(float(m.group(1)),), label=sentence[:200], url=url, page=page, ptype=ptype, snap=snap,
                    effective=None, title=heading, extra=extra))
    return facts


def _term(desc: str):
    m = re.search(r"(\d+)\s*-?\s*(?:months?\b|mo\b)", desc)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*-?\s*years?\b", desc)
    return int(m.group(1)) * 12 if m else None


def _tier(desc: str, family: str):
    # Reading the (kind, low, high) tier a label describes: model years, else a balance or loan amount
    years = _model_years(desc)
    if years:
        return ("model_year",) + years
    amounts = _amounts(desc)
    if amounts:
        return ("balance" if family in DEPOSIT_FAMILIES else "loan_amount",) + amounts
    return "", None, None


def _model_years(desc: str):
    if "model year" not in desc:
        return None
    m = re.search(r"\b(\d{4})-(\d{2,4})\b", desc)
    if m:
        first = int(m.group(1))
        last = int(m.group(2))
        if last < 100:
            # Giving a two-digit end year the start year's century, as in "2024-27"
            last = (first // 100) * 100 + last
        return float(min(first, last)), float(max(first, last))
    m = re.search(r"\b(\d{4})\b", desc)
    if not m:
        return None
    year = float(m.group(1))
    if "older" in desc:
        return None, year
    if "newer" in desc:
        return year, None
    return year, year


def _amounts(desc: str):
    # Reading a balance or loan-amount tier from labels such as "$10000-$24999.99", "$25000+", "$832750 or less"
    if "$" not in desc:
        return None
    num = r"(\d+(?:\.\d+)?)"
    m = re.search(r"\$" + num + r"-\$?" + num, desc)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"\$" + num + r"\s*(?:\+|or more|and up|and above)", desc)
    if m:
        return float(m.group(1)), None
    m = re.search(r"\$" + num + r"\s*or less", desc)
    if m:
        return None, float(m.group(1))
    return None


def _value(cell: str):
    # Reading a cell's rate as shown and as numbers: "5.39%" and (5.39,), or "12.50%-19.50%" and (12.5, 19.5)
    found = PCT.findall(cell)
    if not found:
        return None
    if len(found) == 2 and re.search(r"%\s*(?:-|to)\s*\d", norm(cell)):
        return f"{found[0]}%-{found[1]}%", (float(found[0]), float(found[1]))
    return f"{found[0]}%", (float(found[0]),)


# Reading the non-rate facts: phone and routing numbers, branches, hours, events, claims, figures and links


@dataclass(frozen=True)
class Location:
    name: str
    address: str        # the full line: street, city, state and ZIP
    city: str
    state: str
    zip: str
    amenities: str      # e.g. "Drive Up Window", "24-hour ATM"
    kind: str           # branch, dupaco_atm, office or network
    url: str
    snap: int


@dataclass(frozen=True)
class BranchHours:
    branch: str                 # the town, such as "Platteville"
    address: str
    lobby: tuple                # (days, hours) pairs, such as ("Monday-Thursday", "8:30 a.m.-5 p.m.")
    drive_up: tuple
    url: str
    snap: int
    published: Optional[date]

    def schedule(self, which: str):
        # Joining the "lobby" or "drive_up" hours into one line: "Monday-Thursday 8:30 a.m.-5 p.m.; Friday ..."
        return "; ".join(f"{days} {hours}" for days, hours in getattr(self, which))


@dataclass(frozen=True)
class Event:
    kind: str                   # closure or opening
    when: date
    sentence: str
    url: str
    title: str
    snap: int
    published: Optional[date]


@dataclass(frozen=True)
class Claim:
    kind: str                   # branch_count, members, assets, employees, counties or ordinal
    value: str
    sentence: str               # the quoted sentence; for counts, the whole paragraph
    topic: str                  # the kind, or for ordinals "<series>|<what is counted>"
    url: str
    snap: int
    published: Optional[date]


ROUTING = re.compile(r"routing\s+(?:number|#|no\.?)\D{0,40}?(\d{9})\b", re.I)


def routing_numbers(md: str):
    return ROUTING.findall(md)


PHONE = re.compile(r"(?<![\d-])(?:\(\d{3}\)\s?|\d{3}[-.])\d{3}[-.]\d{4}(?![\d-])")


def phones(md: str):
    # Finding every phone number, however it's written, as ("800-373-7600", the 60 characters before it)
    found = []
    for m in PHONE.finditer(md):
        digits = re.sub(r"\D", "", m.group(0))
        before = clean(strip_links(md[max(0, m.start() - 60):m.start()]))
        found.append((f"{digits[:3]}-{digits[3:6]}-{digits[6:]}", before))
    return found


def locations(md: str, url: str, snap: int):
    # Reading locator entries: a "### Name" heading, then a "street, City, ST, 12345" line and amenities
    lines = [line.strip() for line in md.splitlines()]
    found = []
    for i, line in enumerate(lines):
        heading = re.match(r"^#{2,4}\s+(.+)$", line)
        if not heading:
            continue
        name = clean(heading.group(1))
        address_line = None
        amenities = ""
        for below in lines[i + 1:i + 8]:
            if below.startswith("#"):
                break
            m = re.match(r"^(.+?),\s*([A-Za-z .'-]+),\s*([A-Z]{2}),\s*(\d{5})$", clean(below))
            if m and not address_line:
                address_line = m
            elif address_line and ("·" in below or "ATM" in below or "Drive" in below):
                amenities = clean(below)
        if not address_line:
            continue
        found.append(Location(name=name, address=clean(address_line.group(0)),
                              city=address_line.group(2).strip().title(), state=address_line.group(3),
                              zip=address_line.group(4), amenities=amenities, kind=_location_kind(name),
                              url=url, snap=snap))
    return found


def _location_kind(name: str):
    lowered = name.lower()
    if lowered.startswith("dupaco credit union"):
        return "branch"
    if "dupaco atm" in lowered:
        return "dupaco_atm"
    if lowered.startswith("dupaco"):
        return "office"
    return "network"


_DAYS = r"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?"
_DAY_LINE = re.compile(rf"^({_DAYS}(?:\s*(?:-|through|to|&|and)\s*{_DAYS})?)\s*:\s*(.+)$", re.I)
_SERVICE = re.compile(r"^(lobby|drive[- ]?(?:up|thru|through))(?:\s+hours)?\s*:?\s*", re.I)


def branch_hours(md: str, url: str, snap: int, published: Optional[date], title: str):
    """Lobby and drive-up hours from a branch announcement: "Monday-Thursday: 8:30 a.m.-5 p.m." lines under a
    "Lobby" or "Drive-up" label. Any other line ends a block of hours."""
    lines = [clean(strip_links(line)) for line in md.splitlines() if line.strip()]
    branch, address = _branch_named(" ".join(lines), title)
    blocks = []
    service = None
    current = {"lobby": [], "drive_up": []}
    for line in lines:
        label = _SERVICE.match(line)
        rest = line[label.end():] if label else line
        if label:
            service = "lobby" if label.group(1).lower() == "lobby" else "drive_up"
        day = _DAY_LINE.match(rest)
        if day and service:
            current[service].append((clean(day.group(1)), clean(day.group(2))))
        elif not label and (current["lobby"] or current["drive_up"]):
            blocks.append(current)
            current = {"lobby": [], "drive_up": []}
            service = None
    if current["lobby"] or current["drive_up"]:
        blocks.append(current)
    return [BranchHours(branch, address, tuple(b["lobby"]), tuple(b["drive_up"]), url, snap, published)
            for b in blocks]


def _branch_named(text: str, title: str):
    m = re.search(r"\bThe ([A-Z][\w .'-]{2,40}?) (?:Dupaco )?branch(?: is)? located at ([^,]+)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Falling back to the dateline, as in "PLATTEVILLE, Wis. -"
    dateline = re.search(r"\b([A-Z][A-Z .'-]{2,30}), (?:Iowa|Ill\.|Wis\.|Illinois|Wisconsin)\s?-", text)
    if dateline:
        return dateline.group(1).title().strip(), ""
    return title.strip(), ""


_CLOSE = re.compile(r"\b(will close|will be closing|is closing|closed|closes|closing)\b(?!\s+costs)", re.I)
_OPEN = re.compile(r"\b(will open|opened|opens|grand opening|ribbon cutting)\b", re.I)
_PLACE = re.compile(r"\b(branch|location|office|lobby|drive-?up)\b", re.I)


def events(text: str, url: str, snap: int, published: Optional[date], title: str):
    # Finding dated branch openings and closures, in sentences such as "This location will close March 31, 2026"
    found = []
    for sentence in sentences(text.replace("\n", " ")):
        when = month_date(sentence)
        if not when or not (_PLACE.search(sentence) or _PLACE.search(title)):
            continue
        if _CLOSE.search(sentence):
            found.append(Event("closure", when, sentence[:400], url, title, snap, published))
        elif _OPEN.search(sentence):
            found.append(Event("opening", when, sentence[:400], url, title, snap, published))
    return found


# Listing press-release figures that go stale, as (claim kind, display format, conflict topic)
COUNT_CLAIMS = [("branch_count", "{} branch offices", "number of Dupaco branch offices (branch count, locations)"),
                ("counties", "{} counties", "number of counties Dupaco serves (charter area, who can join)"),
                ("assets", "{}", "Dupaco total assets")]
_ABOUT = [
    ("branch_count", re.compile(r"(\d+)\s+branch(?:es|\s+offices?|\s+office\s+locations?|\s+locations?)\b", re.I)),
    ("members", re.compile(r"(?:membership has grown to|serves|serving)\s+(?:more than|over|nearly)?\s*"
                           r"([\d,]{5,})", re.I)),
    ("assets", re.compile(r"assets\s+(?:exceeding|of more than|of over|over|of)\s+"
                          r"(\$[\d.,]+\s*(?:million|billion))", re.I)),
    ("employees", re.compile(r"(?:over|more than|nearly)?\s*(\d{2,4})\+?\s+employees", re.I)),
    ("counties", re.compile(r"(\d+)\s+counties", re.I)),
]
_ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                  "eleventh", "twelfth"]
_ORDINAL = re.compile(rf"\b({'|'.join(_ORDINAL_WORDS)}|\d+(?:st|nd|rd|th))\s+(time|consecutive year|straight year|"
                      r"year in a row)", re.I)
_AWARD = re.compile(r"\b(recogni[sz]\w*|named|rank\w*|award\w*|honor\w*|best|top)\b", re.I)


def claims(text: str, url: str, page: str, ptype: str, snap: int, published: Optional[date]):
    """Dated statements that go stale: the counts in a press release's closing paragraph about Dupaco ("22 branch
    office locations"), and award ordinals ("the fifth time Forbes recognized Dupaco")."""
    found = []
    text = text.replace("\n", " ")
    start = text.find("member-owned financial cooperative")
    if start >= 0:
        about = text[max(0, start - 80):start + 900]
        for kind, pattern in _ABOUT:
            for m in pattern.finditer(about):
                found.append(Claim(kind, m.group(1).strip(), about[:700], kind, url, snap, published))
    if ptype == "news":
        # Treating numbered copies of one release ("...-3") as one series
        series = re.sub(r"-\d+$", "", page)
        for sentence in sentences(text[:start] if start >= 0 else text):
            m = _ORDINAL.search(sentence)
            # Skipping counts within a short spell, as in "the second time in just a month"
            if m and _AWARD.search(sentence) and "in just" not in sentence:
                counted = "times" if m.group(2).lower() == "time" else "years in a row"
                found.append(Claim("ordinal", m.group(1).lower(), sentence[:400], f"{series}|{counted}", url, snap,
                                   published))
    return found


def ordinal_value(word: str):
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS.index(word) + 1
    m = re.match(r"(\d+)", word)
    return int(m.group(1)) if m else 0


@dataclass(frozen=True)
class Figure:
    # Recording a figure a page states under a label, such as "Members: 182,627" in a newsletter table
    label: str
    value: str
    url: str
    page: str
    snap: int


_FIGURE = re.compile(r"^[|\s*•-]*([A-Z][A-Za-z'&/ ]{2,40}?)\s*[:|]\s*\|?\s*"
                     r"(\$?\d[\d,.]*\s?(?:%|million|billion|thousand)?)\s*\|?\s*$")


def figures(text: str, url: str, page: str, snap: int):
    # Finding "Label: value" lines and two-cell table rows whose value is an amount, a count or a percentage
    found = []
    for line in text.split("\n"):
        m = _FIGURE.match(line.strip())
        # Skipping years, since "Scene in: 2026" is not a figure
        if m and not re.fullmatch(r"(?:19|20)\d\d", m.group(2).strip()):
            found.append(Figure(" ".join(m.group(1).split()), m.group(2).strip(), url, page, snap))
    return found


def links(md: str):
    found = []
    for m in re.finditer(r"(?<!!)\[([^\]\[]{2,80})\]\((https?://[^)\s]+)\)", md):
        anchor = clean(m.group(1)).lower()
        if 0 < len(anchor.split()) <= 8 and not anchor.startswith("!"):
            found.append((anchor, m.group(2)))
    return found
