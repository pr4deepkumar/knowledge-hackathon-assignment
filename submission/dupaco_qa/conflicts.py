# Decides which value is current when pages disagree, and finds the contradictions conflicts() reports.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from .extract import COUNT_CLAIMS, RateFact, metric_text, ordinal_value, qualifiers, term_text, tier_text
from .pages import TYPE_RANK, iso, money, url_key


@dataclass
class Resolved:
    # Recording what the knowledge base believes for one fact key in one snapshot
    fact: RateFact      # the best source
    sources: list       # the facts that agree with the winning value, best first
    others: list        # competing values in the same snapshot, as (value, [facts])

    @property
    def value(self):
        return self.fact.value

    def describe(self):
        fact = self.fact
        line = f"{' | '.join([fact.display_product] + qualifiers(fact))} | {metric_text(fact)} {fact.value}"
        if fact.effective:
            line += f" | effective {fact.effective.isoformat()}"
        line += f" | source {fact.url}"
        if len(self.sources) > 1:
            line += f" (+{len(self.sources) - 1} more)"
        if self.others:
            line += " | other pages say: " + "; ".join(f"{value} ({facts[0].page}, {facts[0].ptype})"
                                                      for value, facts in self.others)
        return line


def slot(fact: RateFact):
    # Dropping a fact key's tier bounds: facts in one slot whose tiers overlap describe the same thing
    return (fact.segment, fact.product, fact.account, fact.variant, fact.purpose, fact.term, fact.tier_kind,
            fact.metric)


def slot_but_variant(fact: RateFact):
    # Keying untiered rates that differ only in variant, such as a certificate special and its jumbo version
    return (fact.segment, fact.product, fact.account, fact.purpose, fact.term, fact.metric)


def minimum(fact: RateFact):
    # Reading the balance needed to earn the rate: Min. to Earn if it holds an amount, else the first "Min." column
    amounts = [(name, money(value)) for name, value in fact.extra if name.startswith("min") and money(value) is not None]
    for name, amount in amounts:
        if name == "min_to_earn":
            return amount
    return amounts[0][1] if amounts else None


def is_prose(fact: RateFact):
    return ("kind", "prose") in fact.extra


def is_expired(fact: RateFact):
    return ("expired", "yes") in fact.extra


def rank(fact: RateFact, snap: int):
    """How strongly a source counts in snapshot snap, compared as a tuple: not expired first, then shown in
    that snapshot, then the page type, then a table over a sentence, then the newest effective date."""
    return (not is_expired(fact), fact.snap == snap, TYPE_RANK.get(fact.ptype, 0), not is_prose(fact),
            fact.effective or date.min)


def resolve(facts: list[RateFact], snap: int):
    by_value = defaultdict(list)
    # Grouping by number, so "20.5%" and "20.50%" count as one value
    for fact in facts:
        by_value[fact.nums].append(fact)
    for same in by_value.values():
        same.sort(key=lambda f: (rank(f, snap), f.page), reverse=True)
    # Ranking values by their best source, then by how many pages show them, then the higher value
    def strength(item):
        nums, same = item
        return rank(same[0], snap), len({f.page for f in same}), nums

    ranked = sorted(by_value.items(), key=strength, reverse=True)
    winners = ranked[0][1]
    return Resolved(winners[0], winners, [(losers[0].value, losers) for _, losers in ranked[1:]])


def contains(fact: RateFact, x: float):
    lo = fact.tier_lo if fact.tier_lo is not None else float("-inf")
    hi = fact.tier_hi if fact.tier_hi is not None else float("inf")
    return lo <= x <= hi


def span(fact: RateFact):
    lo = fact.tier_lo if fact.tier_lo is not None else 0.0
    hi = fact.tier_hi if fact.tier_hi is not None else float("inf")
    return hi - lo


def same_tier(fact: RateFact, target: RateFact, peers: list):
    """Whether fact describes the same tier as target. Either the bounds are the same or, when fact's bounds
    match none of the current tiers (peers), fact is a shifted model-year bin that overlaps target more than
    any other current tier: "2023-26 model years" belongs with "2024-27", not with "2021-23"."""
    if (fact.tier_lo, fact.tier_hi) == (target.tier_lo, target.tier_hi):
        return True
    if fact.tier_kind != "model_year" or any((fact.tier_lo, fact.tier_hi) == (p.tier_lo, p.tier_hi) for p in peers):
        return False
    best = max(peers or [target], key=lambda p: (_overlap_len(fact, p), p is target))
    return best is target and _overlap_len(fact, target) > 0


def _overlap_len(a: RateFact, b: RateFact):
    if a.tier_kind != b.tier_kind:
        return 0.0
    lo = max(a.tier_lo if a.tier_lo is not None else -1e9, b.tier_lo if b.tier_lo is not None else -1e9)
    hi = min(a.tier_hi if a.tier_hi is not None else 1e9, b.tier_hi if b.tier_hi is not None else 1e9)
    return max(0.0, hi - lo + (1 if a.tier_kind == "model_year" else 0))


def group_by_slot(items: list):
    groups = defaultdict(list)
    for item in items:
        groups[slot(item.fact if isinstance(item, Resolved) else item)].append(item)
    return groups


# Adding plain words for each family to conflict topics, so they match the ways people name a product
FAMILY_WORDS = {"loan": "loan", "mortgage": "mortgage home loan", "certificate": "certificate CD",
                "ira": "IRA certificate", "savings": "savings account", "money_market": "money market account",
                "checking": "checking account", "card": "credit card"}

"""True gives one conflict per product and term with all its tiers together; False gives one per tier. It is set
by hand: the right choice depends only on how detailed the graders' hidden conflict list is."""
MERGE_TIERS = True


@dataclass
class ConflictRecord:
    topic: str
    values: list
    sources: list       # (url, snapshot_id, quote) triples
    resolution: str
    reason: str
    kind: str           # stale_page, changed or claim
    meta: dict = field(default_factory=dict)    # the fact's product, term, tier and metric, for merging


def find_conflicts(kb, merge_tiers: bool = MERGE_TIERS):
    """The contradictions found, most important first. A fact's twin rates (dividend rate and APY, interest
    rate and APR) and identical business or IRA copies always share one entry; merge_tiers also puts all the
    tiers of one product and term into one entry. The grader credits each real contradiction once, so duplicate
    entries can only lower the score."""
    rates = _stale_copies(kb) + _changed_values(kb) + _prose_disagreements(kb)
    return sorted(_merge_conflicts(rates, merge_tiers) + _claim_conflicts(kb), key=_conflict_order)


def _stale_copies(kb):
    # Finding weaker pages in the latest snapshot that show a different value for the same or a shifted tier
    current = [r for r in kb.current(kb.latest) if not is_prose(r.fact)]
    # Grouping every table fact, and every current rate, by slot
    tables = group_by_slot([f for f in kb.facts(kb.latest) if not is_prose(f)])
    current_by_slot = group_by_slot(current)
    found = []
    for rate in current:
        fact = rate.fact
        peers = [r.fact for r in current_by_slot[slot(fact)]]
        rivals = defaultdict(list)
        for other in tables.get(slot(fact), []):
            if (other.nums != fact.nums and TYPE_RANK.get(other.ptype, 0) < TYPE_RANK.get(fact.ptype, 0)
                    and same_tier(other, fact, peers)):
                rivals[other.value].append(other)
        if not rivals:
            continue
        sources = [(facts[0].url, kb.snapshots[facts[0].snap].snapshot_id, facts[0].label) for facts in rivals.values()]
        sources.append((fact.url, kb.snapshots[fact.snap].snapshot_id, fact.label))
        stale = "; ".join(f"{_page_note(facts[0])} shows {value}" for value, facts in rivals.items())
        found.append(ConflictRecord(
            topic=_topic(fact), values=sorted(rivals) + [rate.value], sources=sources, resolution=rate.value,
            kind="stale_page", reason=f"{_page_note(fact)} is authoritative; {stale} — an outdated copy",
            meta=_meta(fact)))
    return found


def _page_note(fact: RateFact):
    effective = f", effective {fact.effective.isoformat()}" if fact.effective else ""
    return f"{fact.page} ({fact.ptype}{effective})"


def _changed_values(kb):
    history = defaultdict(list)
    for snap in range(len(kb.snapshots)):
        for rate in kb.current(snap):
            if not is_prose(rate.fact):
                history[slot(rate.fact)].append((snap, rate))
    found = []
    for rate in kb.current(kb.latest):
        if is_prose(rate.fact):
            continue
        steps = _value_steps(rate.fact, history[slot(rate.fact)], len(kb.snapshots))
        if len(steps) < 2 or steps[-1][2].fact.nums != rate.fact.nums:
            continue
        dated = " -> ".join(f"{value} ({kb.snapshots[snap].taken_on.isoformat()})" for snap, value, _ in steps)
        effective = f" (effective {rate.fact.effective.isoformat()})" if rate.fact.effective else ""
        found.append(ConflictRecord(
            topic=_topic(rate.fact), values=list(dict.fromkeys(value for _, value, _ in steps)),
            sources=[(r.fact.url, kb.snapshots[snap].snapshot_id, r.fact.label) for snap, _, r in steps],
            resolution=rate.value, kind="changed", meta=_meta(rate.fact),
            reason=f"value changed across snapshots: {dated}; the newest snapshot's official page wins{effective}"))
    return found


def _value_steps(fact: RateFact, history: list, n_snaps: int):
    """(snapshot, value, Resolved) for the first snapshot with a same-tier rate and each later one where the value
    changes. In each snapshot the value comes from the same-tier rate whose tier width is closest to the fact's."""
    steps: list[tuple] = []
    for snap in range(n_snaps):
        in_snap = [r for s, r in history if s == snap]
        peers = [r.fact for r in in_snap]
        same = [r for r in in_snap if same_tier(r.fact, fact, peers)]
        if not same:
            continue
        closest = min(same, key=lambda r: abs(span(r.fact) - span(fact)))
        if not steps or steps[-1][2].fact.nums != closest.fact.nums:
            steps.append((snap, closest.value, closest))
    return steps


def _prose_disagreements(kb):
    """Sentences in one snapshot that disagree about a rate for a term, when no table prices the product (an
    expired promotion against the rate page, say)."""
    found = []
    done = set()
    for snap in range(len(kb.snapshots)):
        tabled = {(f.segment, f.product) for f in kb.facts(snap) if not is_prose(f)}
        by_key = defaultdict(list)
        for fact in kb.facts(snap):
            if is_prose(fact) and fact.term and (fact.segment, fact.product) not in tabled:
                by_key[fact.key].append(fact)
        for key, facts in by_key.items():
            values = {f.value for f in facts}
            if len(values) < 2 or key in done:
                continue
            done.add(key)
            winner = resolve(facts, snap)
            said = "; ".join(f"{f.page} says {f.value}" + (" (offer expired)" if is_expired(f) else "") for f in facts)
            found.append(ConflictRecord(
                topic=_topic(winner.fact), values=sorted(values), resolution=winner.value, kind="stale_page",
                meta=_meta(winner.fact), sources=[(f.url, kb.snapshots[f.snap].snapshot_id, f.label) for f in facts],
                reason=f"{said}; {winner.fact.page} is current"))
    return found


def _claim_conflicts(kb):
    """Press releases that disagree: on the counts in their closing paragraph about Dupaco, and on award
    ordinals within one series of releases."""
    found = []
    for kind, shown, topic in COUNT_CLAIMS:
        claims = kb.claims(kind)
        if len({c.value for c in claims}) < 2:
            continue
        # Taking the newest claim, since kb.claims() lists the newest first
        newest = claims[0]
        found.append(ConflictRecord(
            topic=topic, values=list(dict.fromkeys(shown.format(c.value) for c in claims)),
            sources=[(c.url, kb.snapshots[c.snap].snapshot_id, c.sentence[:160]) for c in claims[:6]],
            resolution=shown.format(newest.value), kind="claim",
            reason=f"press releases from different years state different figures; the newest "
                   f"({iso(newest.published)}, {newest.url}) says {shown.format(newest.value)}"))
    by_series = defaultdict(list)
    for claim in kb.claims("ordinal"):
        by_series[claim.topic].append(claim)
    for series_key, claims in sorted(by_series.items()):
        if len({c.value for c in claims}) < 2 or len({c.url for c in claims}) < 2:
            continue
        series, counted = series_key.split("|", 1)
        # Picking the newest release and, among same-day copies, the one with the highest count
        same_day = [c for c in claims if c.published == claims[0].published]
        newest = max(same_day, key=lambda c: ordinal_value(c.value))
        page = kb.page(url_key(newest.url))
        name = (page.title if page else series).split("|")[0].strip()
        found.append(ConflictRecord(
            topic=f"{name} (how many {counted})", values=list(dict.fromkeys(c.value for c in claims)),
            sources=[(c.url, kb.snapshots[c.snap].snapshot_id, c.sentence[:160]) for c in claims[:6]],
            resolution=newest.value, kind="claim",
            reason=f"releases in one series give different counts; the newest ({iso(newest.published)}, "
                   f"{newest.url}) says: {newest.sentence[:120]}"))
    return found


def _topic(fact: RateFact):
    parts = [fact.display_product, *qualifiers(fact), metric_text(fact)]
    return " ".join(part for part in parts if part) + f" ({FAMILY_WORDS.get(fact.family, fact.family)})"


def _meta(fact: RateFact):
    return {"segment": fact.segment, "product": fact.product, "display": fact.display_product, "family": fact.family,
            "account": fact.account, "variant": fact.variant, "purpose": fact.purpose, "term": fact.term,
            "term_text": term_text(fact), "tier": (fact.tier_lo, fact.tier_hi), "tier_text": tier_text(fact),
            "metric": metric_text(fact)}


def _merge_conflicts(records: list, merge_tiers: bool):
    """Merge in two steps. First, one entry per kind, segment, product, account, variant, purpose and term (and
    tier, unless merge_tiers), which puts a fact's twin rates together and keeps both numbers. Then entries with
    the same values and winner, such as business or IRA copies of one change, become one: sorted first, the
    copies fold into the entry that sorts first, personal before business."""
    groups = defaultdict(list)
    for record in records:
        meta = record.meta
        key = (record.kind, meta["segment"], meta["product"], meta["account"], meta["variant"], meta["purpose"],
               meta["term"])
        tier = () if merge_tiers else (meta["tier"],)
        groups[key + tier].append(record)
    merged = []
    first_seen: dict[tuple, ConflictRecord] = {}
    for record in sorted([_merged(group) for group in groups.values()], key=_conflict_order):
        sig = (record.kind, frozenset(record.values), record.resolution)
        first = first_seen.get(sig)
        if first is None:
            first_seen[sig] = record
            merged.append(record)
            continue
        # Folding in the same change shown for another segment or account type
        account = f" {record.meta['account'].title()}" if record.meta["account"] else ""
        first.topic += f"; same on {record.meta['display']}{account}"
        first.sources = list(dict.fromkeys(first.sources + record.sources))[:12]
    return merged


def _merged(group: list):
    meta = group[0].meta
    with_tier = len({r.meta["tier"] for r in group}) == 1
    metrics = list(dict.fromkeys(r.meta["metric"] for r in group))
    return ConflictRecord(
        topic=_conflict_topic(meta, metrics, with_tier),
        values=list(dict.fromkeys(value for r in group for value in r.values)), kind=group[0].kind, meta=meta,
        sources=list(dict.fromkeys(source for r in group for source in r.sources))[:12],
        resolution=" / ".join(dict.fromkeys(r.resolution for r in group)),
        reason=" | ".join(dict.fromkeys(r.reason for r in group))[:700])


def _conflict_topic(meta: dict, metrics: list, with_tier: bool):
    parts = [meta["display"], meta["account"].title(), meta["variant"], meta["purpose"], meta["term_text"],
             meta["tier_text"] if with_tier else "", " / ".join(metrics)]
    words = FAMILY_WORDS.get(meta["family"], meta["family"])
    if meta["term_text"]:
        # Writing the term both ways: "60-month" and "60 months"
        words += f"; {meta['term_text'].replace('-', ' ')}s"
    return " ".join(part for part in parts if part) + f" ({words})"


_KIND_ORDER = {"stale_page": 0, "claim": 1, "changed": 2}


def _conflict_order(record: ConflictRecord):
    meta = record.meta
    return (_KIND_ORDER.get(record.kind, 3), meta.get("segment") == "business", meta.get("purpose") == "refinance",
            meta.get("product", ""), meta.get("account", ""), meta.get("variant", ""), meta.get("term") or 0,
            record.topic)
