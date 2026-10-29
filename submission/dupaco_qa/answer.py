# Answers a question from checked data when it can, otherwise with Claude, whose answer is checked first.
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .conflicts import Resolved, group_by_slot, is_prose, same_tier, slot
from .context import HEADLINE, Context, asks_for_rate, build_context, unpriced_types
from .extract import COUNT_CLAIMS, metric_text, qualifiers, term_text, tier_text
from .pages import TYPE_RANK, PageVersion, iso, tokens, url_key
from .store import KnowledgeBase

# Answering from the knowledge base alone, without Claude

# Replying with this whenever the website doesn't answer the question
REFUSAL = "That isn't listed on our website. Please contact Dupaco and a team member can help."

# Matching question words to link text, trying "become a member" links first for join questions
_LINK_TEXT = [("join", ["become a member"]), ("join", ["join"]), ("member", ["become a member"]), ("rate", ["rate"]),
              ("branch", ["branch"]), ("location", ["location"]), ("atm", ["branch"])]

# Ignoring words before "branch" that don't name a branch, as in "Saturday branch hours"
_NOT_A_BRANCH = set("monday tuesday wednesday thursday friday saturday sunday weekend weekday lobby drive-up "
                    "drive-thru drive-through credit union".split())


def fallback_answer(kb: KnowledgeBase, question: str, as_of: Optional[str] = None):
    """Answer without an LLM: from an entity (a link, the branch count, a closure, the routing number or hours),
    else the matched rate facts, else the best passage, else decline. Facts win over an entity for a rate question
    that doesn't ask for a link, and a loan type the facts don't cover is declined as not published.

    'route' says how the answer was found. 'facts', 'entity' and 'not_published' come from checked data and
    can be trusted as they are; 'no_rate', 'passage' and 'nothing' are guesses an LLM should improve on."""
    ctx = build_context(kb, question, as_of, k=3)
    slots = ctx.slots
    if ctx.facts and unpriced_types(slots.text, ctx.facts):
        return _decline("not_published")
    # Skipping entities when the matched facts already answer a plain rate question
    entity = _entity_answer(kb, ctx) if slots.topics or not ctx.facts else None
    # Preferring the entity unless facts answer a rate question, and always giving a link when asked
    if entity and ("link" in slots.topics or not (ctx.facts and asks_for_rate(slots))):
        # Spreading the entity last, so a declined entity keeps its own confident and route
        return {"confident": True, "note": None, "route": "entity", **entity}
    if ctx.facts:
        return _facts_answer(ctx.facts)
    if asks_for_rate(slots):
        # Declining rate questions without facts, as "not published" when the site never prices the loan type
        unpriced = any(gap.startswith("no rate is published for") for gap in ctx.gaps)
        return _decline("not_published" if unpriced else "no_rate")
    return _passage_answer(ctx) or _decline("nothing")


def _decline(route: str):
    return {"text": REFUSAL, "sources": [], "confident": False, "note": None, "route": route}


def _facts_answer(rates: list):
    if len(rates) > 8:
        # Keeping one headline rate per row (APY, mortgage rate or APR) when the list is long
        headline = [r for r in rates if r.fact.metric == HEADLINE.get(r.fact.family, "apy")]
        if headline:
            rates = headline
    rates = rates[:12]
    urls = []
    for rate in rates:
        urls += [fact.url for fact in rate.sources[:3]]
    ignored = [f"{r.fact.display_product}: ignored {', '.join(value for value, _ in r.others)} from outdated pages"
               for r in rates if r.others]
    return {"text": "; ".join(_fact_line(r) for r in rates) + ".", "sources": list(dict.fromkeys(urls))[:5],
            "confident": True, "route": "facts", "note": "; ".join(ignored) or None}


def _fact_line(rate: Resolved):
    fact = rate.fact
    variant = "" if fact.variant == "standard" else fact.variant
    parts = [fact.account.title(), variant, fact.purpose, term_text(fact), tier_text(fact)]
    details = ", ".join(part for part in parts if part)
    name = f"{fact.display_product} ({details})" if details else fact.display_product
    return f"{name}: {fact.value} {metric_text(fact)}"


def _entity_answer(kb: KnowledgeBase, ctx: Context):
    topics = ctx.slots.topics
    snap = ctx.snap
    if "holiday" in topics:
        # Skipping holiday questions, since weekly schedules and closure notices don't cover holidays
        return None
    if "link" in topics:
        for trigger, anchor_words in _LINK_TEXT:
            if trigger in ctx.slots.text:
                targets = kb.link_targets(anchor_words, snap)
                if targets:
                    url = targets[0][0]
                    return {"text": f"You can use {url}", "sources": [url]}
    if "branch_count" in topics:
        claims = kb.claims("branch_count", snap)
        if claims:
            return {"text": f"Dupaco has {claims[0].value} branch office locations.", "sources": [claims[0].url]}
    if "closure" in topics or "still open" in ctx.slots.text:
        closure = _closure_named(kb, ctx)
        if closure:
            return {"text": f"{closure.sentence} ({closure.title.split('|')[0].strip()})", "sources": [closure.url]}
    if "routing" in topics:
        numbers = kb.routing_numbers(snap).most_common(1)
        if numbers:
            number = numbers[0][0]
            # Citing up to six rate or product pages that show the number, or else the contact page
            pages = kb.pages_with(number, snap)[:6] or ["https://www.dupaco.com/contact"]
            return {"text": f"Dupaco's routing number is {number}.", "sources": pages}
    if "hours" in topics or "drive_up" in topics:
        return _hours_answer(kb, ctx)
    return None


# Ignoring the words every closure notice shares, as search tokens, so they don't match a question by themselves
_CLOSURE_WORDS = set(tokens("branch branches dupaco close closed closes closing closure closures"))


def _closure_named(kb: KnowledgeBase, ctx: Context):
    words = set(tokens(ctx.question))
    for event in kb.events(ctx.snap, kind="closure"):
        if words & (set(tokens(event.title + " " + event.sentence)) - _CLOSURE_WORDS):
            return event
    return None


def _hours_answer(kb: KnowledgeBase, ctx: Context):
    hours = kb.branch_hours(ctx.snap)
    key = "drive_up" if "drive_up" in ctx.slots.topics else "lobby"
    what = "Drive-up" if key == "drive_up" else "Lobby"
    name = " ".join(word for word in ctx.slots.named_branch.split() if word not in _NOT_A_BRANCH)
    if name:
        # Answering a named branch only from its own published hours, and declining if it has none
        mine = _hours_for_branch(hours, key, name, ctx.question.lower())
        if not mine:
            return _decline("entity")
        lines = dict.fromkeys(f"{h.branch} location: {h.schedule(key)}" for h in mine)
        return {"text": f"{what} hours at Dupaco's " + " | ".join(lines) + ".",
                "sources": list(dict.fromkeys(h.url for h in mine))}
    schedules = [schedule for schedule in dict.fromkeys(h.schedule(key) for h in hours) if schedule]
    if not schedules:
        return None
    return {"text": f"{what} hours vary by branch. Typical schedules: " + " | ".join(schedules[:3])
                    + ". Which branch would you like?", "sources": [hours[0].url, "https://www.dupaco.com/contact"]}


def _hours_for_branch(hours: list, key: str, name: str, question: str):
    """The hours records for the branch the member named. A town written next to "branch" decides ("the branch
    in Madison", "the Madison, Wisconsin branches"); failing that, a town that is part of the name itself."""
    listed = [h for h in hours if h.branch and getattr(h, key)]
    beside = []
    for h in listed:
        town = re.escape(h.branch.lower())
        pattern = rf"\bbranch(?:es)?\s+(?:in|at|on)\s+(?:the\s+)?{town}\b|\b{town}(?:,\s*[a-z]+)?\s+branch"
        if re.search(pattern, question):
            beside.append(h)
    if beside:
        return beside
    return [h for h in listed if re.search(rf"\b{re.escape(h.branch.lower())}\b", name)]


def _passage_answer(ctx: Context):
    if not ctx.passages:
        return None
    top = ctx.passages[0]
    # Dropping FAQ question lines, so the member's own words in them don't count toward coverage
    body = "\n".join(line for line in top.text.split("\n") if not line.strip().endswith("?"))
    words = set(tokens(ctx.question))
    coverage = len(words & set(tokens(top.heading + " " + body))) / max(1, len(words))
    text = _best_sentences(body, ctx.question)
    if coverage < 0.5 or not text:
        return None
    return {"text": text, "sources": [top.url], "confident": True, "note": None, "route": "passage"}


def _best_sentences(text: str, question: str, n: int = 3):
    """The n sentences that share the most words with the question, in page order, leaving out fragments of 20
    characters or fewer and FAQ questions (sentences ending in "?")."""
    words = set(tokens(question))
    sentences = []
    for piece in re.split(r"(?<=[.!?])\s+|\n", text):
        sentence = piece.strip()
        if len(sentence) > 20 and not sentence.endswith("?"):
            sentences.append(sentence)
    best = sorted(sentences, key=lambda s: -len(words & set(tokens(s))))[:n]
    return " ".join(s for s in sentences if s in best)


# Answering with Claude

# Letting DUPACO_CLAUDE_* environment variables set the model, effort, timeout, debug output and usage log
MODEL = os.getenv("DUPACO_CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("DUPACO_CLAUDE_EFFORT", "low")
TIMEOUT_S = float(os.getenv("DUPACO_CLAUDE_TIMEOUT", "20"))
# Capping each call at 25 seconds, so a slow API never hits the harness's 60-second limit
DEADLINE_S = 25.0
DEBUG = os.getenv("DUPACO_CLAUDE_DEBUG") == "1"
# Logging one JSON line of token usage per call to this file, when it is set
USAGE_LOG = os.getenv("DUPACO_CLAUDE_USAGE_LOG")
HOST = "https://www.dupaco.com"
# Listing the knowledge base's answer routes that come from checked data
SURE_ROUTES = {"facts", "entity", "not_published"}
# Spotting plain rate questions ("what's the APR?"), which rate facts can answer on their own
RATE_WORDS = re.compile(r"\b(rates?|apys?|aprs?|dividends?|interest|yields?|pays?|paying|earns?|earning)\b", re.I)
# Spotting questions about intros, bonuses, requirements, how to do something or why, which need the page itself
NUANCE = re.compile(r"\bintro|\bpromo|\bbonus|\bspecial offer|\bqualif|\brequire|\bminimum|\beligib|"
                    r"\bhow (?:do|can) i\b|\bwhy\b", re.I)
# Spotting two questions in one ("Can I join online, and how much do I deposit?"), which one fact can't answer
COMPOUND = re.compile(r"\?\s*\S.*\?|\s(?:and|also|plus)\s+(?:how|what|when|where|which|who|why|is|are|do|does|did|"
                      r"can|could|will|would)\b", re.I)
# Spotting two different facts asked for at once, such as a routing number and a phone number
_ASKS = [r"\brouting\b", r"\bphone\b|\bcall\b", r"\baddress\b|\blocated\b", r"\bhours?\b",
         r"\blink\b|\bwebsite\b|\burl\b"]

RULES = """You are the member-service assistant for Dupaco Community Credit Union. Answer the member's question using only Dupaco's website, which you get as a checked rate sheet and key facts (below) plus the details found for the question (in the user message).

How to answer:
- Use the current value from the rate sheet or the matching facts. Numbers marked "outdated copies elsewhere" or "other pages say" are stale copies on old pages: never repeat them. Don't mention old or previous figures unless the member asks how something changed.
- For "what changed" or "is it still ..." questions, compare the before and after text in WHAT CHANGED ON MATCHING PAGES and the rate history: say what changed, between which dates, and what it is now. Something that was removed from a page is no longer offered there.
- Answer exactly what was asked: the product, term, tier (balance, loan amount, model year) and rate type (APY, dividend rate, APR, interest rate) the member named. When the member names an amount, term or tier, give only the value that applies to it. If no tier or term was named, give the current values of the main options briefly (a range or a short list). Use personal products unless the member asks about business ones; for "what do you offer" questions list every matching product, business ones included. When the answer refers to an area, a list or conditions the site spells out (who can join, which loans qualify, what a benefit covers), name what they are rather than only referring to them.
- Keep it short: one to three plain sentences that answer every part of the question. No markdown, bullets or headings. For yes/no questions, start with "Yes" or "No".
- Copy numbers, dates, times, names and addresses exactly as written (for example "8:30 a.m.", "5.39%", "11375 Oldenburg Ln."), and use the site's own wording for counts and rankings (for example "the fifth time", "22 branch office locations"). Don't add numbers the member didn't ask for: no other tiers or terms, examples, conversions or comparisons.
- Include a web address only when the member asks for a link or website, or how to do something online (join, apply, log in); then give the full URL.
- Don't ask the member questions or for personal details (ZIP code, member number, account details). If something varies by branch, give the typical schedules and say it varies by branch.
- When press releases or pages from different years disagree, use the newest one.
- Treat the date in the user message as today when deciding whether something has already happened.
- If the member asks about a period the website doesn't cover exactly (for example "last year"), give the most recent figures it does have, each with the year it covers.
- For how-to or service questions the website doesn't spell out step by step, tell the member how to get it done using the contact options in the key facts; that is an answer, so confident is true.
- If the website doesn't publish a rate, fee, figure or detail that was asked for (see "NOT PUBLISHED ON THE SITE", or nothing relevant is given), say it isn't listed on Dupaco's website and suggest contacting Dupaco; set confident to false and use no numbers at all.
- If the details don't answer the question but a page in the SITE MAP (below) probably does, answer as above for now and list up to 3 of those page paths in "look_up"; you will then be shown those pages. Otherwise leave "look_up" empty.
- Speak as Dupaco's assistant. Never mention the rate sheet, facts, passages, snapshots, the site map or "the information provided".

Return JSON: "answer" (what the member reads), "confident" (false only when the website doesn't answer it), "sources" (URLs of the pages your answer relies on, copied from what you were given), "look_up" (site-map paths to read, usually empty)."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer", "confident", "sources", "look_up"],
          "properties": {"answer": {"type": "string"}, "confident": {"type": "boolean"},
                         "sources": {"type": "array", "items": {"type": "string"}},
                         "look_up": {"type": "array", "items": {"type": "string"}}}}

# Limiting how many site-map pages Claude may ask to read when the passages fall short
LOOK_UP_PAGES = 3

_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_URL = re.compile(r"https?://[^\s)\]|,;\"']+")
# Spotting sentences where Claude talks about its inputs, such as "the rate sheet" or "the information provided"
_PROMPT_TALK = re.compile(r"\b(rate sheet|key facts|passages?|snapshots?|context|knowledge base|"
                          r"(?:provided|given) (?:information|details|facts|data)|in these facts)\b", re.I)


def numbers(text):
    return {round(float(n.replace(",", "")), 4) for n in _NUM.findall(text or "")}


def percents(text):
    # Reading only rate-like numbers ("5.34%", "5.34 percent"), so "6 p.m." never matches a stale 6.00%
    return {round(float(n), 4) for n in re.findall(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b)", text or "", re.I)}


def _model_options(model):
    """Request settings per model. Opus and Fable 5.x models always think, with effort as the only control, and
    get the server-side fallback for refusals. Sonnet 5.5 thinks only between tool calls, and there are none
    here; Haiku 5.5 doesn't think."""
    if model.startswith(("claude-opus-5", "claude-fable-5")):
        return {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    if model.startswith("claude-sonnet-5-5"):
        return {"thinking": {"type": "between_tools"}}
    if model.startswith("claude-haiku-5-5"):
        return {"thinking": {"type": "disabled"}}
    return {}


def trust_kb(base, slots, question):
    """Whether the knowledge base's own answer is enough: it comes from checked data, the question asks one
    simple thing and names no branch, and there is nothing to compare."""
    if base["route"] not in SURE_ROUTES or "change" in slots.topics or slots.named_branch:
        return False
    if COMPOUND.search(question) or sum(bool(re.search(pattern, slots.text)) for pattern in _ASKS) > 1:
        return False
    # Trusting rate facts only for plain rate questions, not ones about offers, requirements, how-to or why
    return base["route"] != "facts" or bool(RATE_WORDS.search(slots.text) and not NUANCE.search(slots.text))


def _plain(text):
    text = re.sub(r"(\*\*|__|`|^#+\s*|^\s*[-*•]\s+)", "", text or "", flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def _without_links(text):
    # Removing web addresses and the "see" or "visit" phrases that lead into them
    text = re.sub(r"\s*[-–(,:]*\s*(?:(?:please )?(?:see|visit|go to|at|here)\s*:?\s*)?" + _URL.pattern + r"\)?",
                  "", text).strip()
    return re.sub(r"\s+([.,;!?])", r"\1", text)


class ClaudeAnswerer:
    def __init__(self, kb, api_key=None, workspace_id=None):
        # Importing the SDK here, so answering without a key never needs it installed
        import anthropic
        self.kb = kb
        options = {"timeout": TIMEOUT_S, "max_retries": 1}
        if api_key:
            options["api_key"] = api_key
        if workspace_id:
            # Sending the workspace ID with every request, which a key not tied to one workspace needs
            options["default_headers"] = {"anthropic-workspace-id": workspace_id}
        self.client = anthropic.Anthropic(**options)
        self._pool = ThreadPoolExecutor(max_workers=4)
        # Caching per snapshot: the system prompt, the numbers in it, and the numbers only stale copies show
        self._system = {}
        self._prompt_numbers = {}
        self._stale = {}

    def system_prompt(self, snap):
        """The rules plus the snapshot's rate sheet, key facts and site map. It is the same for every question,
        so the API caches it."""
        if snap not in self._system:
            day = self.kb.snapshots[snap].taken_on.isoformat()
            rate_sheet = rate_sheet_text(self.kb, snap)
            key_facts = key_facts_text(self.kb, snap)
            pages = site_map(self.kb, snap)
            self._system[snap] = (f"{RULES}\n\nRATE SHEET (Dupaco website as of {day}):\n{rate_sheet}"
                                  f"\n\nKEY FACTS (as of {day}):\n{key_facts}"
                                  f"\n\nSITE MAP (pages on the site as of {day}):\n{pages}")
        return self._system[snap]

    def user_message(self, question, as_of, ctx, snap):
        info = self.kb.snapshots[snap]
        return (f"Today's date: {(as_of or info.taken_on.isoformat())[:10]} (the website as it stood on "
                f"{info.taken_on.isoformat()}).\nMember question: {question}\n\n"
                f"DETAILS FOUND FOR THIS QUESTION:\n{ctx.text()}")

    def _ask(self, question, as_of, ctx, snap):
        start = time.time()
        response = self.client.beta.messages.create(
            model=MODEL, max_tokens=4096, **_model_options(MODEL),
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            system=[{"type": "text", "text": self.system_prompt(snap), "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": self.user_message(question, as_of, ctx, snap)}])
        if USAGE_LOG:
            usage = response.usage
            with open(USAGE_LOG, "a") as log:
                log.write(json.dumps({"model": response.model, "seconds": round(time.time() - start, 2),
                                      "input": usage.input_tokens, "output": usage.output_tokens,
                                      "cache_write": usage.cache_creation_input_tokens or 0,
                                      "cache_read": usage.cache_read_input_tokens or 0}) + "\n")
        if response.stop_reason == "refusal":
            raise RuntimeError("declined by the model")
        return json.loads(next(block.text for block in response.content if block.type == "text"))

    def answer(self, question, as_of=None):
        base = fallback_answer(self.kb, question, as_of)
        ctx = build_context(self.kb, question, as_of)
        if trust_kb(base, ctx.slots, question):
            # Using the knowledge base's answer as it is, with no model call
            return base
        snap = ctx.snap
        try:
            start = time.time()
            reply = self._pool.submit(self._ask, question, as_of, ctx, snap).result(timeout=DEADLINE_S)
            # Looking up the pages Claude named only if it declined and there's time left for a second call
            more = []
            if not reply.get("confident", True) and time.time() - start < TIMEOUT_S:
                more = self._look_up(reply, ctx, question, snap)
            if more:
                # Asking once more with the pages Claude named, since search missed the answer
                ctx.passages += more
                reply = self._pool.submit(self._ask, question, as_of, ctx, snap).result(timeout=DEADLINE_S)
            checked, reason = self._check(reply, ctx, question, snap)
        except Exception as e:
            # Falling back on a timeout, or on network, API, refusal or JSON errors
            checked, reason = None, f"error {type(e).__name__}: {e}"
        if checked:
            return checked
        if DEBUG:
            print(f"[claude fallback] {question!r}: {reason}", file=sys.stderr)
        return base

    def _look_up(self, reply, ctx, question, snap):
        """Passages Claude hasn't seen yet from the site-map pages it asked for: the first LOOK_UP_PAGES of them
        that are on the site."""
        on_site = {key for key, version_snap in self.kb.view(snap).items() if version_snap == snap}
        keys = []
        for path in reply.get("look_up") or []:
            key = url_key(path if str(path).startswith("http") else HOST + str(path))
            if key in on_site and key not in keys:
                keys.append(key)
        if not keys:
            return []
        seen = {(p.url, p.heading) for p in ctx.passages}
        return [p for p in self.kb.page_passages(set(keys[:LOOK_UP_PAGES]), question, snap)
                if (p.url, p.heading) not in seen]

    def _check(self, reply, ctx, question, snap):
        # Returning (answer, None) when Claude's reply can be used, else (None, the reason it can't)
        text = _plain(reply.get("answer", ""))
        if not reply.get("confident", True):
            # Keeping an unconfident reply only if it adds no numbers, else using the standard refusal
            usable = text and not (numbers(text) - numbers(question))
            return {"text": text if usable else REFUSAL, "sources": [], "confident": False, "note": None}, None
        kept = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if not _PROMPT_TALK.search(sentence)]
        text = " ".join(kept).strip()
        if not text:
            return None, "only prompt talk"
        if "link" not in ctx.slots.topics and "join" not in ctx.slots.topics:
            text = _without_links(text)
        outdated = self._outdated_claims(question)
        stale = self._stale_in(text, ctx, outdated, snap)
        if stale:
            return None, f"outdated value {sorted(stale)}"
        stale_words = [v for v in outdated if not numbers(v) and re.search(rf"\b{re.escape(v)}\b", text, re.I)]
        if stale_words:
            return None, f"outdated claim {stale_words}"
        allowed = self._allowed_numbers(ctx, question, snap)
        # Allowing small whole numbers anywhere, such as "3 branches" or "12 months"
        unknown = {n for n in numbers(text) if n not in allowed and not (n.is_integer() and n <= 12)}
        if unknown:
            return None, f"numbers not on the site {sorted(unknown)}"
        ignored = [f"ignored outdated {', '.join(value for value, _ in r.others)} for {r.fact.display_product}"
                   for r in ctx.facts[:4] if r.others]
        return {"text": text, "sources": self._sources(reply.get("sources"), ctx, text, snap), "confident": True,
                "note": "; ".join(ignored) or None}, None

    def _outdated_claims(self, question):
        # Finding older press-release figures the question touches, such as "24 branch offices" or "fourth"
        words = set(tokens(question)) - {"dupaco"}
        outdated = []
        for conflict in self.kb.conflicts():
            if conflict.kind == "claim" and words & (set(tokens(conflict.topic)) - {"dupaco"}):
                outdated += [value for value in conflict.values if value != conflict.resolution]
        return outdated

    def _stale_in(self, text, ctx, outdated, snap):
        """The outdated values the text repeats: rates that only old copies on the site still show, and older
        press-release figures written as a phrase ("24 branch offices", so that "24-hour ATM" is fine)."""
        stale = set() if "change" in ctx.slots.topics else percents(text) & self._stale_numbers(snap)
        for value in outdated:
            words = value.split()[:2]
            if numbers(value) and re.search(r"(?<![\d.])" + r"\s+".join(map(re.escape, words)) + r"\b", text, re.I):
                stale |= numbers(value)
        return stale

    def _stale_numbers(self, snap):
        """Values that some rate table on the site shows but that aren't current anywhere: outdated copies on
        calculators, old guides and promo pages, and superseded tiers."""
        if snap not in self._stale:
            current = {round(x, 4) for rate in self.kb.current(snap) for x in rate.fact.nums}
            tabled = {round(x, 4) for fact in self.kb.facts(snap) if not is_prose(fact) for x in fact.nums}
            self._stale[snap] = tabled - current
        return self._stale[snap]

    def _allowed_numbers(self, ctx, question, snap):
        if snap not in self._prompt_numbers:
            self._prompt_numbers[snap] = numbers(self.system_prompt(snap))
        shown = ctx.text()
        allowed = self._prompt_numbers[snap] | numbers(shown) | numbers(question)
        if ctx.slots.amount:
            # Allowing the member's own amount ("$250k") when restated as "$250,000"
            allowed.add(round(ctx.slots.amount, 4))
        if "change" in ctx.slots.topics:
            # Allowing the gap between two rates the site shows, as in "up 0.50 points"
            rates = sorted(percents(shown))
            allowed |= {round(b - a, 4) for i, a in enumerate(rates) for b in rates[i + 1:]}
        return allowed

    def _sources(self, cited, ctx, text, snap):
        """Claude's citations that are pages on the site or at dupaco.com, the matched facts' pages, and rate or
        product pages showing the answer's first two rates or long numbers. Extra sources cost nothing, and the
        right one earns credit."""
        view = self.kb.view(snap)
        sources = []

        def add(url):
            if url.startswith("/"):
                url = HOST + url
            if not url.startswith("http") or url_key(url) in {url_key(u) for u in sources}:
                return
            if url_key(url) in view or "dupaco.com" in url:
                sources.append(url)

        for url in cited or []:
            add(str(url).strip())
        for rate in ctx.facts[:3]:
            for fact in rate.sources[:2]:
                add(fact.url)
        for value in re.findall(r"\d[\d,]*\.\d+%|\d{6,}", text)[:2]:
            for url in self.kb.pages_with(value, snap)[:2]:
                add(url)
        return sources[:8]


# Building the rate sheet, key facts and site map that make up Claude's system prompt


def rate_sheet_text(kb, snap: Optional[int] = None):
    s = kb.snap_or_latest(snap)
    current = kb.current(s)
    # Grouping every table fact by slot, outdated copies included, and the current rates the same way
    tables = group_by_slot([f for f in kb.facts(s) if not is_prose(f)])
    peers = group_by_slot([r.fact for r in current])
    # Grouping rates by product and then by row, where a row holds, say, a dividend rate and its APY
    products = defaultdict(lambda: defaultdict(list))
    for rate in current:
        f = rate.fact
        row = (f.account, f.variant, f.purpose, f.term, f.tier_kind, f.tier_lo, f.tier_hi)
        products[(f.segment, f.display_product, f.family)][row].append(rate)
    lines = []
    for (segment, name, family), rows in sorted(products.items()):
        facts = [rate.fact for rates in rows.values() for rate in rates]
        # Stating once, in the product heading, the page and effective date that most rows share
        page = Counter(f.page for f in facts).most_common(1)[0][0]
        effective = Counter(f.effective for f in facts).most_common(1)[0][0]
        since = f", effective {effective.isoformat()}" if effective else ""
        lines.append(f"### {name} ({segment} {family}; source {page}{since})")
        for rates in rows.values():
            lines.append(_rate_sheet_row(rates, page, effective, tables, peers))
    return "\n".join(lines)


def _rate_sheet_row(rates: list, page: str, effective, tables: dict, peers: dict):
    first = rates[0].fact
    values = ", ".join(f"{metric_text(r.fact)} {r.value}" for r in rates)
    details = ", ".join(qualifiers(first))
    line = f"- {details}: {values}" if details else f"- {values}"
    notes = [f"source {first.page}"] if first.page != page else []
    if first.effective and first.effective != effective:
        notes.append(f"effective {first.effective.isoformat()}")
    extra = dict(first.extra)
    for column in ("min_to_open", "min_to_earn", "min_balance", "max_balance"):
        if extra.get(column) not in (None, "", "-", "—"):
            notes.append(f"{column.replace('_', ' ')} {extra[column]}")
    if notes:
        line += f" ({'; '.join(notes)})"
    stale = _outdated_copies(rates, tables, peers)
    if stale:
        line += f" [outdated copies elsewhere: {', '.join(stale)}]"
    return line


def _outdated_copies(rates: list, tables: dict, peers: dict):
    # Collecting this row's competing values, then what lower-ranked pages show for the same or a shifted tier
    competing = [value for rate in rates for value, _ in rate.others]
    weaker = set()
    for rate in rates:
        fact = rate.fact
        for other in tables.get(slot(fact), []):
            if (other.nums != fact.nums and TYPE_RANK.get(other.ptype, 0) < TYPE_RANK.get(fact.ptype, 0)
                    and same_tier(other, fact, peers[slot(fact)])):
                weaker.add(other.value)
    return list(dict.fromkeys(competing + sorted(weaker)))


# Listing the links Claude is told about, as (what they're for, words in their anchor text)
_KEY_LINKS = [("join / become a member", ["become a member"]), ("join", ["join"]), ("rates", ["rate"]),
              ("find a branch or ATM", ["branch"])]


def key_facts_text(kb, snap: Optional[int] = None):
    # Taking the routing and phone numbers most pages show, and each count from the newest press release
    s = kb.snap_or_latest(snap)
    lines = []
    routing = kb.routing_numbers(s).most_common(1)
    if routing:
        lines.append(f"- Routing number: {routing[0][0]}")
    phone = kb.phone_numbers(s).most_common(1)
    if phone:
        lines.append(f"- Main phone number: {phone[0][0]}")
    contact = kb.page("/contact", s)
    if contact and contact.sections:
        lines.append(f"- How members get help (contact page): {contact.sections[0].text[:450]}")
    for kind, shown, topic in COUNT_CLAIMS:
        claims = kb.claims(kind, s)
        if claims:
            lines.append(_newest_claim_line(claims[0], shown, topic))
    branches = kb.locations(s, kind="branch")
    if branches:
        lines.append("- Branch addresses shown on the locator page (not every branch is listed): "
                     + "; ".join(loc.address for loc in branches))
    lines += _branch_hours_lines(kb.branch_hours(s))
    for event in kb.events(s):
        lines.append(f"- Branch {event.kind} {event.when.isoformat()}: {event.sentence} "
                     f"({event.title.split('|')[0].strip()})")
    gone = kb.discontinued(s)
    if gone:
        lines.append("- No longer offered (dropped from the official rate pages): " + ", ".join(gone))
    for need, words in _KEY_LINKS:
        urls = [url for url, _ in kb.link_targets(words, s)[:2]]
        if urls:
            lines.append(f"- Link for {need}: {', '.join(urls)}")
    return "\n".join(lines)


def _newest_claim_line(newest, shown: str, topic: str):
    # Quoting the newest release's own sentence, so Claude has the site's exact wording
    said = next((s for s in re.split(r"(?<=[.!?])\s+", newest.sentence) if newest.value in s), "")
    return (f"- {topic.split(' (')[0].capitalize()}: {shown.format(newest.value)} (newest press release, "
            f"{iso(newest.published)}, says: \"{said[:220]}\"; older releases give outdated figures)")


def _branch_hours_lines(hours: list):
    if not hours:
        return []
    lines = ["- Branch hours from branch-opening announcements (hours vary by branch):"]
    seen = set()
    for h in hours:
        schedule = f"lobby {h.schedule('lobby') or 'not given'} | drive-up {h.schedule('drive_up') or 'not given'}"
        if (h.branch, schedule) not in seen:
            seen.add((h.branch, schedule))
            lines.append(f"  - {h.branch}: {schedule}")
    return lines


def site_map(kb, snap: Optional[int] = None, types=("product", "rates", "promo", "guide", "legal", "doc")):
    """One line per official page on the site in snapshot snap, with its title and section headings: what Claude
    may ask to read when search finds nothing, since a member's words can differ from the site's ("credit-builder
    loan" against a "Build Credit and Save" section). Headings found on many pages are left out."""
    s = kb.snap_or_latest(snap)
    pages = []
    for key, version_snap in sorted(kb.view(s).items()):
        version = kb.versions[key][version_snap]
        if version_snap == s and version.ptype in types:
            pages.append(version)
    on_pages = Counter(heading for version in pages for heading in _headings(version))
    lines = []
    for version in pages:
        title = version.title.split("|")[0].strip()
        own = [h for h in _headings(version) if on_pages[h] < 4 and len(h) < 70][:12]
        line = version.key if not title or title.startswith("http") else f"{version.key} — {title}"
        lines.append(line + (": " + "; ".join(own) if own else ""))
    return "\n".join(lines)


def _headings(version: PageVersion):
    # Skipping PDFs, whose headings aren't worth listing
    if version.ptype == "doc":
        return []
    return list(dict.fromkeys(section.heading.split(" > ")[-1] for section in version.sections[1:]))
