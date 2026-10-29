# The knowledge base: every snapshot's pages, facts, search index and change log, queryable as of any snapshot.
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

from . import extract
from .conflicts import (MERGE_TIERS, Resolved, contains, find_conflicts, group_by_slot, is_expired, is_prose,
                        minimum, rank, resolve, slot, slot_but_variant, span)
from .extract import RateFact, metric_text, prose_facts, table_facts, term_text, tier_text
from .pages import (SEARCH_WEIGHT, TYPE_RANK, BM25Index, PageVersion, SectionDoc, boilerplate_lines, clean,
                    clean_sections, find_tables, page_type, publish_date, segment as page_segment, strip_links,
                    tokens, url_key)

# Reading rates from these page types' sentences too, not just their tables
PROSE_TYPES = {"rates", "product", "promo", "guide", "calculator"}
# Capping how many sections search returns from any one page
PAGE_CAP = 3

"""Page types whose changes in wording matter to members. News only counts when a release is added or taken
down, and blogs and calculators change tags, bylines and widgets on every crawl."""
CHANGE_TYPES = {"rates", "product", "promo", "guide", "legal", "doc"}
# Reading labelled figures from these page types, not from the worked examples on calculators and blogs
FIGURE_TYPES = CHANGE_TYPES | {"news"}
# Treating rate sheets and product pages, rank 4 and up, as the official pages
_OFFICIAL_RANK = TYPE_RANK["product"]


@dataclass
class SnapshotInfo:
    idx: int
    snapshot_id: str
    taken_on: date


@dataclass
class Passage:
    url: str
    title: str
    heading: str
    text: str
    ptype: str
    published: Optional[date]
    score: float
    also: list = field(default_factory=list)    # other URLs showing the very same section


@dataclass
class Change:
    # Recording one difference on a page between two consecutive snapshots, searchable like a section
    page: str
    url: str
    title: str
    heading: str
    ptype: str
    snap: int                   # the snapshot the change shows up in
    before: str                 # the date of the earlier snapshot
    after: str                  # the date of the newer one
    removed: list               # lines only in the earlier version (sentences, for PDFs)
    added: list                 # lines only in the newer version
    kind: str = "reworded"      # reworded, page added or page removed
    snaps: set = field(default_factory=set)     # the snapshots it was logged in, kept by BM25Index.add

    @property
    def text(self):
        return "\n".join(self.removed + self.added)


class KnowledgeBase:
    def __init__(self):
        self.snapshots: list[SnapshotInfo] = []
        self.versions: dict[str, dict[int, PageVersion]] = defaultdict(dict)
        self.index = BM25Index()
        # Indexing the change log, which "what changed" questions search
        self.change_index = BM25Index()
        # Keeping each (page, snapshot) version's rate facts, and its other facts by kind
        self._facts: dict[tuple, list[RateFact]] = {}
        self._entities: dict[tuple, dict] = {}
        # Mapping each snapshot to {page key: the snapshot whose version of the page is shown}
        self._views: dict[int, dict[str, int]] = {}
        # Keeping each snapshot's boilerplate lines, in the form the change log compares
        self._boiler: dict[int, set[str]] = {}
        self._cache: dict[tuple, object] = {}

    def add_snapshot(self, snapshot):
        snap = len(self.snapshots)
        self.snapshots.append(SnapshotInfo(snap, snapshot.snapshot_id, date.fromisoformat(snapshot.taken_on[:10])))
        boiler = boilerplate_lines([p.markdown for p in snapshot.pages])
        crawled = set()
        # Treating "/x", "/x/" and "/x#y" as one page, which its shortest URL stands for
        for page in sorted(snapshot.pages, key=lambda p: (url_key(p.url), len(p.url))):
            key = url_key(page.url)
            if key not in crawled:
                crawled.add(key)
                self._add_page(key, page, snap, boiler)
        self._boiler[snap] = _boilerplate_units(boiler)
        if snap > 0:
            self._log_changes(snap, crawled)
        previous = self._views.get(snap - 1, {})
        # Carrying forward the pages this snapshot lacks, from the last snapshot that had them
        self._views[snap] = {**previous, **{key: snap for key in crawled}}
        self._cache.clear()

    def _add_page(self, key: str, page, snap: int, boiler: set):
        ptype = page_type(key)
        version = PageVersion(key=key, url=page.url, title=page.title, snap=snap, ptype=ptype,
                              segment=page_segment(key), published=publish_date(key, ptype, page.markdown),
                              markdown=page.markdown)
        if ptype != "listing":
            version.sections = clean_sections(page.title, page.markdown, boiler)
        self.versions[key][snap] = version
        self._facts[(key, snap)] = _page_rate_facts(version)
        self._entities[(key, snap)] = _page_entities(version)
        for section in version.sections:
            self.index.add(SectionDoc(key, page.url, page.title, section.heading, section.text, ptype,
                                      version.published), snap)

    def _log_changes(self, snap: int, crawled: set):
        before = self.snapshots[snap - 1].taken_on.isoformat()
        after = self.snapshots[snap].taken_on.isoformat()
        boiler = self._boiler[snap - 1] | self._boiler[snap]
        for change in _page_changes(self.versions, self._views[snap - 1], crawled, snap, boiler, before, after):
            self.change_index.add(change, snap)

    @property
    def latest(self):
        return len(self.snapshots) - 1

    def snap_or_latest(self, snap: Optional[int]):
        return self.latest if snap is None else snap

    def snap_for(self, as_of: Optional[str]):
        # Finding the newest snapshot taken on or before as_of, or the first one if as_of predates them all
        if not as_of:
            return self.latest
        try:
            day = date.fromisoformat(str(as_of)[:10])
        except ValueError:
            return self.latest
        taken = [s.idx for s in self.snapshots if s.taken_on <= day]
        return taken[-1] if taken else 0

    def view(self, snap: Optional[int] = None):
        # Returning {page key: the snapshot of the version that stands for the page as of snap}
        return self._views.get(self.snap_or_latest(snap), {})

    def page(self, key: str, snap: Optional[int] = None):
        version_snap = self.view(snap).get(key)
        return self.versions[key][version_snap] if version_snap is not None else None

    def facts(self, snap: Optional[int] = None):
        s = self.snap_or_latest(snap)
        if ("facts", s) not in self._cache:
            facts = []
            for key, version_snap in sorted(self.view(s).items()):
                facts += self._facts.get((key, version_snap), [])
            self._cache[("facts", s)] = facts
        return self._cache[("facts", s)]

    def current(self, snap: Optional[int] = None):
        # Resolving each fact key's value as of snap, then dropping what's discontinued, ended or outranked
        s = self.snap_or_latest(snap)
        if ("current", s) in self._cache:
            return self._cache[("current", s)]
        by_key = defaultdict(list)
        for fact in self.facts(s):
            by_key[fact.key].append(fact)
        resolved = [resolve(facts, s) for facts in by_key.values()]
        # Noting which products a table prices, and the strongest page type pricing each in this snapshot
        strongest = defaultdict(int)
        in_tables = set()
        for rate in resolved:
            product = (rate.fact.segment, rate.fact.product)
            if not is_prose(rate.fact):
                in_tables.add(product)
                if rate.fact.snap == s and not is_expired(rate.fact):
                    strongest[product] = max(strongest[product], TYPE_RANK.get(rate.fact.ptype, 0))
        dropped = self._discontinued(s)
        kept = []
        for rate in resolved:
            fact = rate.fact
            product = (fact.segment, fact.product)
            if product in dropped or is_expired(fact):
                # Skipping products the official pages stopped listing, and offers that ended
                continue
            if is_prose(fact) and product in in_tables:
                # Skipping sentences about a product that a table already prices
                continue
            if not is_prose(fact) and TYPE_RANK.get(fact.ptype, 0) < min(strongest[product], _OFFICIAL_RANK):
                # Skipping tables from pages weaker than the best one pricing the product, but never official ones
                continue
            kept.append(rate)
        kept.sort(key=lambda r: (r.fact.segment, r.fact.product, r.fact.account, r.fact.variant, r.fact.purpose,
                                 r.fact.term or 0, r.fact.tier_kind, r.fact.tier_lo or 0, r.fact.metric))
        self._cache[("current", s)] = kept
        return kept

    def _official_products(self, s: int):
        # Collecting the (segment, product) pairs a rate sheet or product page in snapshot s states
        if ("official", s) not in self._cache:
            self._cache[("official", s)] = {(f.segment, f.product) for f in self.facts(s)
                                            if f.snap == s and not is_expired(f)
                                            and TYPE_RANK.get(f.ptype, 0) >= _OFFICIAL_RANK}
        return self._cache[("official", s)]

    def _discontinued(self, s: int):
        # Finding products the official pages listed in an earlier snapshot but no longer list in snapshot s
        earlier = set().union(*[self._official_products(t) for t in range(s)]) if s else set()
        return earlier - self._official_products(s)

    def discontinued(self, snap: Optional[int] = None):
        return sorted(product for _, product in self._discontinued(self.snap_or_latest(snap)))

    def catalog(self, snap: Optional[int] = None):
        groups = defaultdict(list)
        for rate in self.current(snap):
            groups[(rate.fact.segment, rate.fact.display_product)].append(rate.fact)
        products = []
        for (segment, name), facts in sorted(groups.items()):
            products.append({
                "product": name, "segment": segment, "family": facts[0].family,
                "metrics": sorted({metric_text(f) for f in facts}),
                "terms": sorted({term_text(f) for f in facts if f.term}, key=_leading_number),
                "tiers": sorted({tier_text(f) for f in facts if f.tier_kind}),
                "variants": sorted({v for f in facts for v in (f.account, f.variant, f.purpose) if v}),
                "pages": sorted({f.url for f in facts})[:4],
            })
        return products

    def lookup(self, product=None, snap: Optional[int] = None, *, segment: Optional[str] = None,
               term: Optional[int] = None, amount: Optional[float] = None, year: Optional[int] = None,
               account: Optional[str] = None, variant: Optional[str] = None, purpose: Optional[str] = None,
               metric=None):
        """Current facts matching the filters. The product and the metric may each be a name or a list of names.

        With a year or an amount, tiers from different pages that contain it compete as one fact, such as a
        calculator's "2023-26 model year" bin and the rate sheet's "2024-27" bin for a 2025 car."""
        s = self.snap_or_latest(snap)
        names = {p.lower() for p in ([product] if isinstance(product, str) else product or [])}
        metrics = {metric} if isinstance(metric, str) else set(metric or [])

        def matches(fact: RateFact):
            if names and fact.product.lower() not in names and fact.display_product.lower() not in names:
                return False
            if segment and fact.segment != segment:
                return False
            if metrics and fact.metric not in metrics:
                return False
            if term is not None and fact.term != term:
                return False
            if account is not None and fact.account != account:
                return False
            if variant is not None and fact.variant != variant:
                return False
            if purpose is not None and fact.purpose != purpose:
                return False
            if year is not None and not (fact.tier_kind == "model_year" and contains(fact, year)):
                return False
            if amount is not None:
                if fact.tier_kind in ("balance", "loan_amount"):
                    return contains(fact, amount)
                if fact.tier_kind == "model_year" and year is not None:
                    # Letting the year pick a model-year tier, which no amount or minimum applies to
                    return True
                # Checking the amount against the minimum column instead, and dropping facts that have none
                need = minimum(fact)
                return need is not None and amount >= need
            return True

        rates = [r for r in self.current(s) if matches(r.fact)]
        # Preferring personal products over business ones when both match, unless the question names a segment
        if segment is None and any(r.fact.segment == "personal" for r in rates):
            rates = [r for r in rates if r.fact.segment == "personal"]
        if year is None and amount is None:
            return rates
        rates = self._tightest_tiers(rates, matches, s)
        if amount is not None:
            rates = _highest_minimum(rates)
        return rates

    def _tightest_tiers(self, rates: list, matches, s: int):
        """One answer per slot: the tightest tier that contains the point ("$200,000 or less" for a $150,000 loan),
        with every other page's value for that point, such as a calculator's older bin, as a competitor."""
        best_rates = []
        for fact_slot, group in sorted(group_by_slot(rates).items(), key=lambda item: str(item[0])):
            tightest = min(span(r.fact) for r in group)
            best = max((r for r in group if span(r.fact) == tightest), key=lambda r: rank(r.fact, s))
            keys = {r.fact.key for r in group}
            others = defaultdict(list)
            for fact in self.facts(s):
                # Counting the best tier's other values and non-current tiers as competitors, not other current tiers
                if (slot(fact) == fact_slot and matches(fact) and fact.nums != best.fact.nums
                        and (fact.key == best.fact.key or fact.key not in keys)):
                    others[fact.nums].append(fact)
            competitors = [(facts[0].value, facts) for _, facts in sorted(others.items())]
            best_rates.append(Resolved(best.fact, best.sources, competitors))
        return best_rates

    def pages_with(self, text: str, snap: Optional[int] = None, types=("rates", "product")):
        # Finding the URLs of pages of these types that contain the text, rate sheets first
        hits = []
        for key, version_snap in sorted(self.view(snap).items()):
            version = self.versions[key][version_snap]
            if version.ptype in types and text in version.markdown:
                hits.append((-TYPE_RANK.get(version.ptype, 0), version.url))
        return [url for _, url in sorted(hits)]

    def search(self, query: str, snap: Optional[int] = None, k: int = 8, types: Optional[Iterable[str]] = None,
               pages: Optional[set] = None):
        s = self.snap_or_latest(snap)
        view = self.view(s)
        taken = self.snapshots[s].taken_on if self.snapshots else date.today()
        types = set(types) if types else None

        def allow(doc: SectionDoc):
            if SEARCH_WEIGHT.get(doc.ptype, 0) == 0:
                return False
            if (types and doc.ptype not in types) or (pages and doc.page not in pages):
                return False
            # Keeping only the sections in each page's version as of s (doc.snaps lists the snapshots that had them)
            return view.get(doc.page) in doc.snaps

        def boost(doc: SectionDoc):
            weight = SEARCH_WEIGHT.get(doc.ptype, 0.5)
            if doc.published:
                # Discounting older news and blog posts by 5% a year, down to 60%
                age = max(0, (taken - doc.published).days) / 365.0
                weight *= max(0.6, 1.0 - 0.05 * age)
            if view.get(doc.page) != s:
                # Lowering pages that are no longer on the site
                weight *= 0.8
            return weight

        passages = []
        per_page = Counter()
        by_text = {}
        for score, doc in self.index.search(query, k=k * 4, allow=allow, boost=boost):
            if doc.text in by_text:
                # Showing a section that appears at several URLs once, listing the other URLs on it
                if doc.url not in by_text[doc.text].also:
                    by_text[doc.text].also.append(doc.url)
            elif per_page[doc.page] < PAGE_CAP:
                # Taking a few sections per page, so one page can't crowd out the rest
                per_page[doc.page] += 1
                by_text[doc.text] = Passage(doc.url, doc.title, doc.heading, doc.text, doc.ptype, doc.published,
                                            round(score, 3))
                passages.append(by_text[doc.text])
            if len(passages) == k:
                break
        return passages

    def changes(self, query: str, snap: Optional[int] = None, k: int = 4):
        """Change-log entries that match the query, up to snapshot snap, best first and at most two per page.
        Removed text is searchable too, so "is X still offered?" finds the change where X went away."""
        s = self.snap_or_latest(snap)
        found = []
        per_page = Counter()
        shown = set()
        hits = self.change_index.search(query, k=k * 10, allow=lambda c: c.snap <= s,
                                        boost=lambda c: SEARCH_WEIGHT.get(c.ptype, 0.5))
        for _, change in hits:
            # Showing the same change at two URLs only once
            sig = (change.kind, change.snap, change.text)
            if per_page[change.page] < 2 and sig not in shown:
                per_page[change.page] += 1
                shown.add(sig)
                found.append(change)
            if len(found) == k:
                break
        return found

    def page_passages(self, keys: set, query: str, snap: Optional[int] = None, max_chars: int = 3500):
        """Most of each of these pages, about max_chars a page: the sections that match the query best first, then
        the rest in page order, and the first always goes in however long. Claude asks for these pages when the
        passages found don't answer the question."""
        s = self.snap_or_latest(snap)
        order = {(p.url, p.heading, p.text): i for i, p in enumerate(self.search(query, s, k=60, pages=keys))}
        passages = []
        for key in sorted(keys):
            version = self.page(key, s)
            if not version:
                continue
            used = 0
            sections = sorted(version.sections, key=lambda sec: order.get((version.url, sec.heading, sec.text),
                                                                          len(order)))
            for section in sections:
                if used and used + len(section.text) > max_chars:
                    # Skipping a section too long to add now, since a shorter one may still fit
                    continue
                used += len(section.text)
                passages.append(Passage(version.url, version.title, section.heading, section.text, version.ptype,
                                        version.published, 0.0))
        return passages

    def _entities_in(self, kind: str, snap: Optional[int] = None):
        found = []
        for key, version_snap in sorted(self.view(snap).items()):
            found += self._entities.get((key, version_snap), {}).get(kind, [])
        return found

    def routing_numbers(self, snap: Optional[int] = None):
        return Counter(self._entities_in("routing", snap))

    def phone_numbers(self, snap: Optional[int] = None):
        return Counter(number for number, _ in self._entities_in("phones", snap))

    def locations(self, snap: Optional[int] = None, kind: Optional[str] = None):
        seen = set()
        found = []
        for location in self._entities_in("locations", snap):
            if (kind is None or location.kind == kind) and location.address not in seen:
                seen.add(location.address)
                found.append(location)
        return found

    def branch_hours(self, snap: Optional[int] = None):
        return self._entities_in("hours", snap)

    def events(self, snap: Optional[int] = None, kind: Optional[str] = None):
        return [e for e in self._entities_in("events", snap) if kind is None or e.kind == kind]

    def claims(self, kind: Optional[str] = None, snap: Optional[int] = None):
        # Listing dated claims, newest publication first
        found = [c for c in self._entities_in("claims", snap) if kind is None or c.kind == kind]
        return sorted(found, key=lambda c: (c.published or date.min, c.url), reverse=True)

    def figures(self, query: str, snap: Optional[int] = None, k: int = 3):
        """Labelled figures whose label shares the query's rarer words ("how many members" finds "Members"), each
        with every value a page states for it, newest document first, as (label, [(value, url, on the site
        since)])."""
        s = self.snap_or_latest(snap)
        stated = self._entities_in("figures", s)
        labels = {f.label.lower() for f in stated}
        # Counting in how many labels each word appears
        df = Counter(word for label in labels for word in set(tokens(label)))
        words = set(tokens(query)) - {"dupaco"}
        score = {label: _label_score(label, words, df) for label in labels}
        found = []
        for label in sorted((x for x in labels if score[x] >= 0.5), key=lambda x: (-score[x], x))[:k]:
            # Mapping each (value, url) to the snapshot in which its document first appeared
            first_seen = {}
            for f in stated:
                if f.label.lower() == label:
                    first_seen.setdefault((f.value, f.url), min(self.versions[f.page]))
            newest_first = sorted(first_seen.items(), key=lambda item: (-item[1], item[0]))
            shown = next(f.label for f in stated if f.label.lower() == label)
            found.append((shown, [(value, url, self.snapshots[i].taken_on.isoformat())
                                  for (value, url), i in newest_first]))
        return found

    def link_targets(self, words: Iterable[str], snap: Optional[int] = None):
        """URLs that product, rate and promo pages link to with anchor text containing all the words, as (url, score)
        pairs, best first. A target crawled in any snapshot up to snap counts double, even if it was taken down."""
        view = self.view(snap)
        words = [w.lower() for w in words]
        score = Counter()
        for (anchor, url), n in Counter(self._entities_in("links", snap)).items():
            if all(w in anchor for w in words):
                score[url] += n * (2 if url_key(url) in view else 1)
        return score.most_common()

    def conflicts(self, merge_tiers: bool = MERGE_TIERS):
        if ("conflicts", merge_tiers) not in self._cache:
            self._cache[("conflicts", merge_tiers)] = find_conflicts(self, merge_tiers)
        return self._cache[("conflicts", merge_tiers)]


def _page_rate_facts(version: PageVersion):
    if version.ptype in ("listing", "legal"):
        return []
    where: dict = dict(url=version.url, page=version.key, ptype=version.ptype, segment=version.segment,
                       snap=version.snap, page_title=version.title)
    facts = []
    for table in find_tables(version.markdown):
        facts += table_facts(table, **where)
    if version.ptype in PROSE_TYPES:
        facts += prose_facts(version.markdown, **where)
    return facts


def _page_entities(version: PageVersion):
    """A page version's other facts by kind. Routing numbers, phones and locations are read from every page,
    hours from any page that mentions a lobby, and the rest only from the page types that carry them."""
    md = version.markdown
    text = "\n".join(section.text for section in version.sections)
    url = version.url
    snap = version.snap
    found = {"routing": extract.routing_numbers(md), "phones": extract.phones(md),
             "locations": extract.locations(md, url, snap), "hours": [], "events": [], "claims": [], "links": [],
             "figures": []}
    if "lobby" in md.lower():
        found["hours"] = extract.branch_hours(md, url, snap, version.published, version.title)
    if version.ptype in ("news", "blog", "product"):
        found["events"] = extract.events(text, url, snap, version.published, version.title)
    if version.ptype in ("news", "blog"):
        found["claims"] = extract.claims(text, url, version.key, version.ptype, snap, version.published)
    if version.ptype in ("product", "rates", "promo"):
        found["links"] = extract.links(md)
    if version.ptype in FIGURE_TYPES:
        found["figures"] = extract.figures(text, url, version.key, snap)
    return found


def _highest_minimum(rates: list):
    """Of untiered rates that differ only in variant, the one with the highest minimum, which the amount has
    already met: $150,000 earns the jumbo special ($100,000 minimum), not the $1,000 one."""
    top = defaultdict(float)
    for rate in rates:
        if not rate.fact.tier_kind:
            key = slot_but_variant(rate.fact)
            top[key] = max(top[key], minimum(rate.fact))
    return [r for r in rates if r.fact.tier_kind or minimum(r.fact) == top[slot_but_variant(r.fact)]]


def _label_score(label: str, words: set, df: Counter):
    """How well a figure's label matches the query's words, the rarer the shared words the better. A word in many
    labels ("fee") counts for little, and the score is zero unless three quarters of the label's own words are in
    the query: "net worth ratio" finds "Regulatory Net Worth Ratio", but "late fee" is not the "safe deposit box
    past due late fee"."""
    own = set(tokens(label)) - {"dupaco"}
    if not own or len(own & words) / len(own) < 0.75:
        return 0
    return sum(1 / df[w] for w in sorted(words & set(tokens(label))))


def _leading_number(text: str):
    m = re.match(r"(\d+)", text)
    return int(m.group(1)) if m else 0


"""The change log: what each page added, dropped or reworded between consecutive snapshots. It is searchable,
so a question about something that went away still finds the text the page used to have."""
# Matching CAPTCHA widget text
_NOT_CONTENT = re.compile(r"select all (?:images|squares)|click verify|click skip", re.I)
# Matching link targets and line breaks, which aren't part of the wording
_LINK_TAIL = re.compile(r"\]\([^)]*\)|\(?https?://\S+|<br\s*/?>", re.I)


def _page_changes(versions: dict, previous: dict, crawled: set, snap: int, boiler: set, before: str, after: str):
    """The changes since the snapshot before, pages added or taken down included. previous is the page view as of
    that snapshot, and boiler is the boilerplate of both snapshots."""
    gone = {key for key, s in previous.items() if s == snap - 1 and key not in crawled}
    log = []
    for key in sorted(crawled | gone):
        old = versions[key][previous[key]] if key in previous else None
        new = versions[key].get(snap)
        version = new or old
        ptype = version.ptype
        # Logging news only when a release is added or taken down
        news_release = ptype == "news" and not (old and new)
        if ptype not in CHANGE_TYPES and not news_release:
            continue
        old_units = _units(old, boiler) if old else []
        new_units = _units(new, boiler) if new else []
        if old and new:
            kind = "reworded"
            hunks = _diff(old_units, new_units)
        else:
            kind = "page added" if new else "page removed"
            # Logging the first three lines of a page that came or went, since they say what it was about
            hunks = [(version.title, old_units[:3], new_units[:3])] if old_units or new_units else []
        for heading, removed, added in hunks:
            log.append(Change(key, version.url, version.title, heading, version.ptype, snap, before, after,
                              [text for _, text in removed], [text for _, text in added], kind))
    return _without_template_changes(log)


def _diff(old: list, new: list):
    matcher = difflib.SequenceMatcher(None, [text for _, text in old], [text for _, text in new], autojunk=False)
    hunks = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "equal":
            heading = new[j1][0] if j2 > j1 else old[i1][0]
            hunks.append((heading, old[i1:i2], new[j1:j2]))
    return hunks


def _without_template_changes(log: list):
    """The log without template lines, and without rewordings left with no real change. A line added or removed on
    many pages at once is a change to the site's template, such as a disclaimer or a footer; a rate row
    repeated on five pages is still a real change."""
    page_lines = {(c.page, line) for c in log if c.kind == "reworded" for line in c.removed + c.added}
    pages = Counter(line for _, line in page_lines)
    template = {line for line, n in pages.items() if n >= 10}
    kept = []
    for change in log:
        if change.kind == "reworded":
            change.removed = [line for line in change.removed if line not in template]
            change.added = [line for line in change.added if line not in template]
            # Skipping changes that only touched template lines, punctuation or formatting
            if _words(change.removed) == _words(change.added):
                continue
        kept.append(change)
    return kept


def _units(version: PageVersion, boiler: set):
    """A page version as the (heading, text) units to compare: lines, or sentences for PDFs, whose line breaks
    move between crawls. Boilerplate, widget text and fragments of fewer than three words ("Apply",
    "32 Results") are left out; short rate rows ("60 Months | 5.39% APR") stay."""
    units = []
    for section in version.sections:
        if version.ptype == "doc":
            pieces = re.split(r"(?<=[.!?])\s+", section.text.replace("\n", " "))
        else:
            pieces = section.text.split("\n")
        for line in map(_unit_text, pieces):
            n_words = len(re.findall(r"[A-Za-z]{2,}", line))
            enough = n_words >= 3 or n_words and re.search(r"\d\s?%|\$\s?\d", line)
            if enough and line not in boiler and not _NOT_CONTENT.search(line):
                units.append((section.heading, line))
    return units


def _boilerplate_units(boiler: set):
    return {_unit_text(clean(strip_links(line))) for line in boiler}


def _unit_text(line: str):
    return " ".join(_LINK_TAIL.sub(" ", line).split())


def _words(lines: list):
    return re.findall(r"[a-z0-9]+", " ".join(lines).lower())
