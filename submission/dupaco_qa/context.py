# Reads a member's question and gathers what the knowledge base knows about it.
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .conflicts import Resolved, is_prose
from .extract import match_product, metric_text, qualifiers, term_text, tier_text
from .pages import MONTHS, focus, norm, tokens
from .store import KnowledgeBase, Passage

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20,
                 "thirty": 30, "forty": 40, "sixty": 60}

# Rewriting the member's wording as the site's before the product is matched
_ALIASES = [
    (r"\bcds?\b|\bshare certificates?\b|\bcertificate of deposits?\b|\bcertificates? of accounts?\b", "certificate"),
    (r"\b(car|truck|suv|used car|new car)s?\b", "auto"),
    (r"\bhome equity line\b", "heloc"),
    (r"\bfirst rate checking\b", "1st rate checking"),
    (r"\bjumbo special\b", "jumbo certificate special"),
    (r"\brefis?\b", "refinance"),
]

_MONTHS = "|".join(MONTHS)

# Matching what comes before a year that belongs to a date ("july 31, 2026", "in 2026") rather than a car
_DATE_BEFORE = re.compile(rf"(?:\b(?:{_MONTHS})\.?(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?,?|\d{{1,2}}/(?:\d{{1,2}}/)?|"
                          r"\b(?:as of|in|since|of|during|by|until|through|from|before|after|early|mid|late))[\s-]*$")

# Listing words that mean a whole product family, unless a specific product or term is named too
_FAMILY_WIDE = [("card", r"credit cards?\b|\bcards\b"), ("checking", r"\bchecking\b"),
                ("certificate", r"\bcertificates?\b"), ("savings", r"\bsavings\b"), ("money_market", r"money markets?")]
# Mapping a bare family word to the product it stands for
_FAMILY_DEFAULT = {"certificate": "Certificate"}

_MORTGAGE_WORDS = re.compile(r"\b(?:fixed|refi|refinanc\w*|purchase|house|home)\b")
_OTHER_FAMILY = re.compile(r"\b(?:cds?|certificates?|ira|auto|car|equity|heloc|improvement|boat|rv|personal|student|"
                           r"business|card)\b")

_HOLIDAYS = (r"\bholidays?\b|\bthanksgiving\b|\bchristmas\b|\bnew year|\beaster\b|\bmemorial day\b|\blabor day\b|"
             r"\bindependence day\b|\bfourth of july\b|\bjuly 4(?:th)?\b|\bjuneteenth\b|\bveterans day\b|"
             r"\bcolumbus day\b|\bpresidents'? day\b|\bmartin luther king\b|\bmlk\b")

# Matching hours as an amount ("how many hours did staff volunteer?"), not opening hours
_HOURS_SPENT = re.compile(r"\bhow many\b[^?]*\bhours?\b|"
                          r"\bhours?\b[^?]*\b(?:spent|logged|volunteer\w*|worked|donated)\b|"
                          r"\bvolunteer\w*\s+hours?\b|\btotal\b[^?]*\bhours?\b")

_DAYS = (r"(?:today|tomorrow|now|late|early|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|"
         r"sundays?|weekends?)")
# Listing what a question can be about besides rates; "hours" leaves out "24 hours"
_TOPICS = [
    ("hours", rf"(?<!\d )(?<!\d-)\bhours?\b|\bwhat time\b|\bstill open\b|"
              rf"\bopens? (?:at \d|on |until|till|{_DAYS})|\bclosed? (?:on |{_DAYS})"),
    ("drive_up", r"drive[- ]?(up|thru|through)"),
    ("address", r"\baddress\b|\blocated\b|\bwhere is\b"),
    ("routing", r"\brouting\b"),
    ("phone", r"\bphone\b|\bcall\b|\bnumber\b"),
    ("link", r"\blink\b|\burl\b|\bwebsite\b|\bonline\b"),
    ("branch_count", r"how many branch|number of branch|how many locations"),
    ("closure", r"\bclos(e|ed|ing|ure)\b"),
    ("join", r"\bjoin\b|\bmember(ship)?\b|\beligib"),
    ("change", r"\bchang|\bbefore\b|\bprevious|\bused to\b|\bincrease|\bdecrease|\bwent (up|down)\b|"
               r"\bgo(ne)? (up|down)\b|\braised?\b|\blowered\b|\bhistory\b|\bsince\b|\bdifferent\b|"
               r"\b(?:went|gone|go) away\b|\bdropped\b|\bdiscontinu(?:ed|ing)\b|\breplaced\b|"
               r"\brolled out\b|\bintroduced\b|\blaunched\b|\badded\b|\bno longer\b|\banymore\b|"
               r"\bstill\b|\b(?:higher|lower)\b[^?]*\bthan\b|\b(?:was|were)\b[^?]*\b(?:now|today|currently)\b"),
    ("holiday", _HOLIDAYS),
]

# Ignoring words before "branch" that aren't part of a branch's name
_NOT_A_NAME = set("""a an the your our my this that these each every all any which what when where how is are
was do does did can could will would of at for in to on near nearest closest local main same different one
many new dupaco dupaco's full-service""".split())

# Matching a date written in the question: "May 21, 2026", "mid-July 2026", "the end of June", "7/1/2026"
_WRITTEN_DATE = re.compile(rf"\b(?:(early|beginning of|start of|mid|middle of|late|end of)[\s-]+)?"
                           rf"({_MONTHS}|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?"
                           r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(20\d\d))?\b"
                           r"|\b(\d{1,2})/(\d{1,2})/(20\d\d|\d\d)\b", re.I)
_MONTH_NUM = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
# Mapping each part of a month to its day; "end of" and a month with no day, such as "May 2026", mean its last day
_PART_DAY = {"early": 5, "beginning of": 1, "start of": 1, "mid": 15, "middle of": 15, "late": 25}

# Matching a term, or terms sharing a unit: "60-month", "5 years", "15-, 20- and 30-year", "12 to 24 months"
_TERMS = re.compile(r"\b((?:\d{1,3}\s*-?\s*(?:,|and|or|&|to)\s*)*\d{1,3})\s*-?\s*(months?|mo|years?|yr)\b(?!\s*old)")


@dataclass
class Slots:
    # Recording what a question asks about
    text: str                                       # the question as _normalized() rewrote it
    product: str = ""
    family: str = ""
    segment: Optional[str] = None
    term: Optional[int] = None                      # set when the question names exactly one term
    terms: list = field(default_factory=list)       # every term named, in months
    year: Optional[int] = None
    amount: Optional[float] = None
    variant: Optional[str] = None
    account: Optional[str] = None
    purpose: Optional[str] = None
    metric: list = field(default_factory=list)
    topics: list = field(default_factory=list)      # what it asks about besides rates: hours, address, ...
    family_wide: str = ""                           # "card" for "what credit cards do you offer?"
    named_branch: str = ""                          # "galena" for "the Galena branch"

    def filters(self):
        # Returning the slots that narrow a rate lookup, as lookup() keywords
        filters = {"segment": self.segment, "term": self.term, "year": self.year, "amount": self.amount,
                   "variant": self.variant, "account": self.account, "purpose": self.purpose,
                   "metric": self.metric or None}
        return {name: value for name, value in filters.items() if value is not None}


def read_question(question: str):
    text = _normalized(question)
    slots = Slots(text=text)
    slots.product, slots.family, _ = match_product([text])
    if "business" in text:
        slots.segment = "business"
    slots.terms = _terms(text)
    slots.term = slots.terms[0] if len(slots.terms) == 1 else None
    if not slots.product and _sounds_like_mortgage(text, slots.terms):
        slots.product, slots.family, _ = match_product([text + " mortgage"])
    slots.year = _model_year(text, slots.family)
    slots.amount = _amount(text)
    slots.variant = _variant(text, slots.family)
    slots.account = next((a for a in ("roth", "traditional") if a in text), None)
    if slots.family == "mortgage" and "refinanc" in text:
        slots.purpose = "refinance"
    elif slots.family == "mortgage" and slots.product.startswith("Home Loan"):
        slots.purpose = "purchase"
    slots.family_wide = _family_wide(text, slots)
    slots.metric = _metrics(text, slots.family)
    slots.topics = _topics(text)
    slots.named_branch = _named_branch(question)
    return slots


def question_date(kb, question: str):
    """The date the question itself asks about, as YYYY-MM-DD: "On May 21, 2026, what were ...", "back in
    mid-July, what did ... pay". Only questions in the past tense or that say "as of" have one. Questions
    about now ("is it higher now than on May 19?") and about what changed don't: their date is the one to
    compare with. Dates before the first snapshot are ignored."""
    text = norm(question)
    if not kb.snapshots or not re.search(r"\b(?:was|were|did|had)\b|\bas of\b", text):
        return None
    if re.search(r"\b(?:now|today|currently)\b", text) or "change" in read_question(question).topics:
        return None
    first = kb.snapshots[0].taken_on
    this_year = kb.snapshots[-1].taken_on.year
    for m in _WRITTEN_DATE.finditer(text):
        day = _written_date(m, this_year)
        if day and day >= first:
            return day.isoformat()
    return None


def asks_for_rate(slots: Slots):
    return bool(slots.metric) or bool(re.search(r"\b(rates?|apy|apr|dividend|interest)\b", slots.text))


def _normalized(question: str):
    text = norm(question)
    for word, number in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(number), text)
    for pattern, site_word in _ALIASES:
        text = re.sub(pattern, site_word, text)
    return text


def _terms(text: str):
    # Finding every term the question names, in months and in order: "the 10-month and the 17-month" gives [10, 17]
    terms = []
    for m in _TERMS.finditer(text):
        months_per_unit = 12 if m.group(2).startswith("y") else 1
        for n in map(int, re.findall(r"\d+", m.group(1))):
            # Skipping year counts over 50, since no term runs that long
            if (months_per_unit == 1 or n <= 50) and n * months_per_unit not in terms:
                terms.append(n * months_per_unit)
    return terms


def _sounds_like_mortgage(text: str, terms: list):
    """Members name a mortgage by its term and type ("a 30-year fixed", "15-year refi") without saying
    "mortgage". Other products with 15, 20 or 30-year terms (boat and RV, home equity, home improvement) are
    named when they are meant."""
    if not terms or not set(terms) <= {180, 240, 360}:
        return False
    return bool(_MORTGAGE_WORDS.search(text)) and not _OTHER_FAMILY.search(text)


def _model_year(text: str, family: str):
    # Finding the car's model year in a loan question: a year that is neither part of a date nor a tier label
    if family not in ("loan", ""):
        return None
    for m in re.finditer(r"\b(19[5-9]\d|20[0-4]\d)\b", text):
        # Skipping a year range such as "2023-26 model years", which is a tier label, not the member's car
        if re.match(r"\s*[-–]\s*\d{2,4}\b", text[m.end():]):
            continue
        if not _DATE_BEFORE.search(text[max(0, m.start() - 20):m.start()]):
            return int(m.group(1))
    return None


def _amount(text: str):
    m = re.search(r"\$\s?(\d+(?:\.\d+)?)\s*(k|thousand|million)?\b", text)
    if not m:
        return None
    scale = {"k": 1e3, "thousand": 1e3, "million": 1e6}.get(m.group(2) or "", 1)
    return float(m.group(1)) * scale


def _variant(text: str, family: str):
    non_jumbo = re.compile(r"\bnon-?\s?jumbo\b")
    jumbo = re.search(r"\bjumbo\b", non_jumbo.sub("", text))
    standard = non_jumbo.search(text) or re.search(r"\b(standard|regular)\b", text)
    # Choosing neither variant when both are named, as in "standard and jumbo"
    if jumbo and not standard:
        return "jumbo"
    if standard and not jumbo and family in ("certificate", "ira") and "special" in text:
        return "standard"
    return None


def _family_wide(text: str, slots: Slots):
    """The product family the question asks to list ("what credit cards do you offer?"), when it names the
    family rather than one product or term."""
    family = next((f for f, pattern in _FAMILY_WIDE if re.search(pattern, text)), "")
    wants_list = re.search(r"\b(rates?|apy|apr|dividend|interest|offer|options|types|kinds|list|all)\b", text)
    if not family or not wants_list:
        return ""
    bare_family_word = slots.product == _FAMILY_DEFAULT.get(family) and slots.term is None and "special" not in text
    return family if not slots.product or bare_family_word else ""


def _metrics(text: str, family: str):
    metrics = []
    if re.search(r"\bapy\b|annual percentage yield|\byield\b", text):
        metrics.append("apy")
    if re.search(r"\bdividend", text):
        metrics.append("dividend_rate")
    if re.search(r"\bapr\b|annual percentage rate", text):
        metrics.append("apr")
        # Asking for both the rate and the APR when a mortgage question names both
        if family == "mortgage" and re.search(r"\brates?\b", text.replace("annual percentage rate", "")):
            metrics.append("rate")
    return metrics


def _topics(text: str):
    topics = [topic for topic, pattern in _TOPICS if re.search(pattern, text)]
    if "hours" in topics and _HOURS_SPENT.search(text):
        topics.remove("hours")
    return topics


def _named_branch(question: str):
    """A proper name right before "branch" in the member's own wording ("the Galena Branch", "Hy-Vee branch"),
    in lower case."""
    m = re.search(r"((?:[A-Z][\w'.&-]*\s+){1,3})[Bb]ranch(?:es)?\b", question or "")
    if not m:
        return ""
    return " ".join(w for w in m.group(1).split() if w.lower() not in _NOT_A_NAME).lower()


def _written_date(m: re.Match, default_year: int):
    """The date a _WRITTEN_DATE match names, or None. A month alone ("may", "march") is not a date; a month
    without a day ("May 2026", "the end of May") means its last day unless the part of the month says otherwise."""
    part, month, day, year, slash_month, slash_day, slash_year = m.groups()
    try:
        if month:
            if not (day or year or part):
                return None
            y = int(year or default_year)
            mon = _MONTH_NUM[month[:3]]
            return date(y, mon, int(day) if day else _PART_DAY.get(part, calendar.monthrange(y, mon)[1]))
        return date(int(slash_year) + (2000 if len(slash_year) == 2 else 0), int(slash_month), int(slash_day))
    except ValueError:
        return None


# Listing loan types the site may not price; if no published rate names one, the matched facts are for another loan
_LOAN_TYPES = [
    (r"\badjustable\b|\barms?\b|\b\d+/\d+\b", "adjustable-rate (ARM) loans"),
    (r"\bconstruction\b", "construction loans"),
    (r"\bfha\b", "FHA loans"),
    (r"\bva\b|\bveterans affairs\b", "VA loans"),
    (r"\busda\b", "USDA loans"),
    (r"\bballoon\b", "balloon loans"),
    (r"\binterest-only\b", "interest-only loans"),
    (r"\breverse mortgage\b", "reverse mortgages"),
    (r"\bland\b|\blot loan", "land/lot loans"),
]
# Matching dates such as "7/1/2026" or "on 7/1", whose slashes would otherwise read as an ARM term like "5/1"
_SLASH_DATES = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:on|as of|since|by|before|after|from|until|through)\s+"
                          r"\d{1,2}/\d{1,2}\b")

# Picking the one rate to show per product when space is short (deposit families use APY)
HEADLINE = {"mortgage": "rate", "loan": "apr", "card": "apr"}

# Spotting questions that ask for a count or an amount, which a page's labelled figures may answer
_QUANTITY = re.compile(r"\bhow (?:many|much)\b|\bnumber of\b|\btotal\b|\bcount\b|\bratio\b|\bfigures?\b|\bpercent|"
                       r"\bamount\b|\blatest\b|\bfees?\b")

_GENERIC_PRODUCT_WORDS = {"loan", "program", "rate", "account"}
_LINK_WORDS = ("join", "rate", "location", "branch", "apply", "login")


@dataclass
class Context:
    # Collecting everything relevant to one question, as of one snapshot
    question: str
    snap: int
    snapshot_id: str
    taken_on: str
    slots: Slots
    facts: list[Resolved] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    discontinued: list = field(default_factory=list)
    gaps: list = field(default_factory=list)              # what the question asks for that the site doesn't publish
    earlier: list[Passage] = field(default_factory=list)  # the first snapshot's passages, for "what changed" questions
    earlier_date: str = ""
    changes: list = field(default_factory=list)           # the matching change-log entries, for the same questions
    figures: list = field(default_factory=list)           # stated figures as (label, values), for "how many" questions

    def text(self):
        out = [f"Website snapshot: {self.snapshot_id} (as of {self.taken_on})"]
        if self.gaps:
            out += ["", "NOT PUBLISHED ON THE SITE:"]
            out += [f"- {gap}" for gap in self.gaps]
        if self.facts:
            out += ["", "MATCHING RATE FACTS (authoritative value first; 'other pages say' = outdated copies):"]
            out += [f"- {rate.describe()}" for rate in self.facts]
        if self.discontinued:
            out += ["", "NO LONGER LISTED ON OFFICIAL PAGES: " + ", ".join(self.discontinued)]
        for name, lines in self.entities.items():
            if lines:
                out += ["", f"{name.upper()}:"]
                out += [f"- {line}" for line in lines]
        if self.figures:
            out += ["", "FIGURES PAGES STATE (newest document first; an older document may show an earlier figure):"]
            for label, values in self.figures:
                stated = [f"{value} ({url}, on the site since {since})" for value, url, since in values[:4]]
                out.append(f"- {label}: " + "; ".join(stated))
        if self.passages:
            out += ["", "PASSAGES:", passages_text(self.passages, max_chars=1100, query=self.question)]
        if self.changes:
            out += ["", "WHAT CHANGED ON MATCHING PAGES "
                        "(between two website snapshots; 'before' is no longer current):"]
            for change in self.changes:
                out.append(f"- {change.kind} between {change.before} and {change.after}: {change.url} "
                           f"({change.ptype}) — {change.heading}")
                if change.kind != "page added":
                    before = focus("\n".join(change.removed), self.question, 400)
                    out.append("  before: " + (before or "(not on the page)"))
                if change.kind != "page removed":
                    after = focus("\n".join(change.added), self.question, 400)
                    out.append("  after: " + (after or "(removed from the page)"))
        if self.earlier:
            out += ["", f"THE SAME TOPIC AS THE SITE STOOD ON {self.earlier_date} (for comparison only, not current):",
                    passages_text(self.earlier, max_chars=900, query=self.question)]
        return "\n".join(out)


def build_context(kb: KnowledgeBase, question: str, as_of: Optional[str] = None, k: int = 5):
    snap = kb.snap_for(as_of)
    slots = read_question(question)
    info = kb.snapshots[snap]
    ctx = Context(question, snap, info.snapshot_id, info.taken_on.isoformat(), slots)
    ctx.facts = _matching_facts(kb, slots, snap)
    ctx.gaps = _gaps(kb, slots, snap, ctx.facts)
    ctx.discontinued = _discontinued_asked_about(kb.discontinued(snap), question, slots)
    ctx.passages = _passages(kb, question, slots, snap, k)
    if _QUANTITY.search(slots.text):
        ctx.figures = kb.figures(question, snap)
    if "change" in slots.topics and snap > 0:
        # Adding the site as it first stood and the change log, so the answer can say what changed
        ctx.earlier = kb.search(question, 0, k=3)
        ctx.earlier_date = kb.snapshots[0].taken_on.isoformat()
        ctx.changes = kb.changes(question, snap)
    ctx.entities = _entities(kb, slots, snap, ctx.facts)
    return ctx


def lookup_relaxed(kb: KnowledgeBase, slots: Slots, snap: int):
    """Look the product up with the question's filters, dropping only the ones that don't change which fact is
    meant (the rate type, purchase or refinance). A term, tier or jumbo/standard the member named is never
    dropped: if that exact product isn't published, nothing matches and the answer should decline."""
    if not slots.product:
        return []
    filters = slots.filters()
    for dropped in ([], ["metric"], ["metric", "purpose"]):
        kept = {name: value for name, value in filters.items() if name not in dropped}
        if len(slots.terms) > 1:
            # Looking terms up one at a time, since slots.term is unset when several are named
            rates = []
            for term in slots.terms:
                rates += kb.lookup(slots.product, snap, **{**kept, "term": term})
        else:
            rates = kb.lookup(slots.product, snap, **kept)
        if rates:
            return rates
    return []


def _matching_facts(kb: KnowledgeBase, slots: Slots, snap: int):
    if not slots.family_wide:
        return lookup_relaxed(kb, slots, snap)
    # Listing every rate in the family asked about, as in "What credit cards do you offer?", personal ones first
    metrics = set(slots.metric)
    rates = []
    for rate in kb.current(snap):
        fact = rate.fact
        if fact.family == slots.family_wide and not is_prose(fact) and (not metrics or fact.metric in metrics):
            rates.append(rate)
    return sorted(rates, key=lambda rate: rate.fact.segment != "personal")


def unpriced_types(question_text: str, rates: list):
    # Finding the loan types the question asks about that none of these rates mention
    asked = _SLASH_DATES.sub(" ", question_text)
    mentioned = " ".join(f"{r.fact.product} {r.fact.label} {r.fact.title}".lower() for r in rates)
    return [name for pattern, name in _LOAN_TYPES
            if re.search(pattern, asked) and not re.search(pattern, mentioned)]


def _gaps(kb: KnowledgeBase, slots: Slots, snap: int, facts: list):
    # Saying outright what isn't published, so the answer declines instead of reusing a nearby fact
    gaps = [f"no rate is published for {name}" for name in unpriced_types(slots.text, kb.current(snap))]
    if slots.product and not facts and not slots.family_wide:
        gaps.append(_unpublished_product(kb, slots, snap))
    return gaps


def _unpublished_product(kb: KnowledgeBase, slots: Slots, snap: int):
    asked = ", ".join(f"{name}={value}" for name, value in slots.filters().items() if name != "metric")
    gap = f"no published rate for {slots.product} with {asked or 'these details'}"
    offered = next((p for p in kb.catalog(snap) if p["product"].lower() == slots.product.lower()), None)
    if not offered:
        return gap + "; this product has no published rates"
    options = offered["terms"] + offered["tiers"] + offered["variants"]
    return gap + f"; published options: {', '.join(options)}"


def _discontinued_asked_about(products: list, question: str, slots: Slots):
    words = set(tokens(question))
    asked = []
    for name in products:
        if words & (set(tokens(name)) - _GENERIC_PRODUCT_WORDS) or (slots.product and slots.product in name):
            asked.append(name)
    return asked


def _passages(kb: KnowledgeBase, question: str, slots: Slots, snap: int, k: int):
    passages = kb.search(question, snap, k=k)
    branch = slots.named_branch
    if branch:
        # Adding passages that name the branch asked about, often the press release announcing it
        seen = {(p.url, p.heading) for p in passages}
        named = []
        for p in kb.search(f"{question} {branch} branch located", snap, k=20):
            if branch in (p.title + " " + p.text).lower() and (p.url, p.heading) not in seen:
                named.append(p)
        passages += named[:2]
    return passages


def _entities(kb: KnowledgeBase, slots: Slots, snap: int, facts: list):
    # Gathering facts for the question's topics under the headings Claude sees, with rate history for "what changed"
    topics = slots.topics
    found = {}
    if "hours" in topics or "drive_up" in topics:
        lines = [_hours_line(hours) for hours in kb.branch_hours(snap)]
        found["branch hours (from branch announcements)"] = _unique_lines(lines)
    if {"address", "hours", "closure"} & set(topics) or "branch" in slots.text:
        branches = kb.locations(snap, kind="branch")
        found["branch locations (locator)"] = [f"{loc.address} [{loc.amenities}]" for loc in branches]
        found["branch closures / openings"] = [f"{e.kind} {e.when}: {e.sentence} ({e.title}, {e.url})"
                                               for e in kb.events(snap)]
    if "routing" in topics:
        counts = kb.routing_numbers(snap).most_common(2)
        found["routing number"] = [f"{number} (on {n} pages)" for number, n in counts]
    if "phone" in topics:
        counts = kb.phone_numbers(snap).most_common(3)
        found["phone numbers"] = [f"{number} (on {n} pages)" for number, n in counts]
    if "branch_count" in topics:
        claims = kb.claims("branch_count", snap)[:4]
        found["branch count claims (newest first)"] = [f"{c.value} branch offices (published {c.published}, {c.url})"
                                                       for c in claims]
    if "link" in topics or "join" in topics:
        word = next((w for w in _LINK_WORDS if w in slots.text), None)
        if word:
            targets = kb.link_targets([word], snap)[:4]
            found["links"] = [f"{url} ({n} links)" for url, n in targets]
    if "change" in topics:
        found["how the matching rates changed over time"] = [_history_line(kb, rate, snap) for rate in facts[:6]]
        if slots.product and snap > 0:
            found[f"{slots.product} rates in each website snapshot (oldest first)"] = _timeline(kb, slots, snap)
    return found


def _hours_line(hours):
    lobby = hours.schedule("lobby") or "n/a"
    drive_up = hours.schedule("drive_up") or "n/a"
    return f"{hours.branch}: lobby {lobby} | drive-up {drive_up} (published {hours.published}, {hours.url})"


def _unique_lines(lines: list):
    # Dropping lines that repeat an earlier one apart from their "(published ...)" ending
    seen = set()
    unique = []
    for line in lines:
        core = line.split(" (published")[0]
        if core not in seen:
            seen.add(core)
            unique.append(line)
    return unique


def _timeline(kb: KnowledgeBase, slots: Slots, upto: int):
    """The product's headline rates in each snapshot so far, so terms or tiers that appeared, went away or were
    relabelled stand out."""
    runs = []
    for info in kb.snapshots[:upto + 1]:
        rates = kb.lookup(slots.product, info.idx, segment=slots.segment, account=slots.account)
        headline = [r for r in rates if r.fact.metric == HEADLINE.get(r.fact.family, "apy")]
        if headline:
            rates = headline
        rates.sort(key=lambda r: (r.fact.purpose, r.fact.term or 0, r.fact.tier_lo or 0, r.fact.variant, r.fact.metric))
        line = "; ".join(f"{_short_label(r)} {r.value} {metric_text(r.fact)}" for r in rates[:30])
        _add_to_runs(runs, line or "not listed", info.taken_on.isoformat())
    return [f"{_span(first, last)}: {line}" for line, first, last in runs]


def _history_line(kb: KnowledgeBase, rate: Resolved, upto: int):
    fact = rate.fact
    runs = []
    for info in kb.snapshots[:upto + 1]:
        rates = kb.lookup(fact.product, info.idx, segment=fact.segment, term=fact.term, account=fact.account,
                          variant=fact.variant, purpose=fact.purpose, metric=fact.metric)
        same = _rates_for_tier(rates, fact)
        _add_to_runs(runs, same[0].value if same else "not listed", info.taken_on.isoformat())
    label = " ".join(x for x in (fact.display_product, *qualifiers(fact), metric_text(fact)) if x)
    return label + ": " + ", ".join(f"{value} ({_span(first, last)})" for value, first, last in runs)


def _rates_for_tier(rates: list, fact):
    """The rates for the fact's own tier or, failing that, model-year tiers that overlap it: the year bins
    move as new model years come out."""
    exact = [r for r in rates if (r.fact.tier_lo, r.fact.tier_hi) == (fact.tier_lo, fact.tier_hi)]
    if exact or fact.tier_kind != "model_year" or fact.tier_hi is None:
        return exact
    overlapping = []
    for rate in rates:
        other = rate.fact
        if other.tier_kind != "model_year" or other.tier_lo is None or other.tier_lo > fact.tier_hi:
            continue
        if other.tier_hi is not None and fact.tier_lo is not None and other.tier_hi < fact.tier_lo:
            continue
        overlapping.append(rate)
    return overlapping


def _add_to_runs(runs: list, value: str, day: str):
    # Extending the last [value, first day, last day] run if the value is unchanged, else starting a new one
    if runs and runs[-1][0] == value:
        runs[-1][2] = day
    else:
        runs.append([value, day, day])


def _span(first: str, last: str):
    return first if first == last else f"{first} to {last}"


def _short_label(rate: Resolved):
    fact = rate.fact
    parts = [fact.variant, fact.purpose, term_text(fact), tier_text(fact)]
    return " ".join(part for part in parts if part) or fact.display_product


def passages_text(passages: list[Passage], max_chars: int = 1200, query: Optional[str] = None):
    blocks = []
    for i, passage in enumerate(passages, 1):
        published = f", published {passage.published.isoformat()}" if passage.published else ""
        also = f" (also at {', '.join(passage.also[:2])})" if passage.also else ""
        body = focus(passage.text, query, max_chars) if query else passage.text[:max_chars]
        blocks.append(f"[{i}] {passage.url}{also} ({passage.ptype}{published}) — {passage.heading}\n{body}")
    return "\n\n".join(blocks)
