# Entry point for the harness: build() returns the object whose ingest(), answer() and conflicts() it calls.
import os
import sys
from pathlib import Path

"""Claude settings: paste an API key between the quotes to answer with Claude, and a workspace ID if the key
needs one, but never commit a real key. Left empty, the ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) and
ANTHROPIC_WORKSPACE_ID environment variables are used. With no key at all, or with DUPACO_NO_LLM=1, every
answer comes from the knowledge base alone."""
ANTHROPIC_API_KEY = ""
ANTHROPIC_WORKSPACE_ID = ""

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
try:
    # Importing the harness's contract, which is on the path when the harness runs the submission
    from contract import Answer, Conflict, Snapshot, Source  # noqa: F401
except ImportError:
    # Finding the kit's harness folder above the submission, when solution.py is imported outside the harness
    harness = next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())
    sys.path.insert(0, str(harness))
    from contract import Answer, Conflict, Snapshot, Source  # noqa: F401

from dupaco_qa.answer import REFUSAL, ClaudeAnswerer, fallback_answer  # noqa: E402
from dupaco_qa.context import question_date  # noqa: E402
from dupaco_qa.store import KnowledgeBase  # noqa: E402


def _claude(kb):
    # Building the Claude answerer, or returning None when DUPACO_NO_LLM=1 or no key is set
    if os.getenv("DUPACO_NO_LLM") == "1":
        return None
    key = ANTHROPIC_API_KEY.strip()
    if not (key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        return None
    workspace = ANTHROPIC_WORKSPACE_ID.strip() or os.getenv("ANTHROPIC_WORKSPACE_ID") or None
    try:
        return ClaudeAnswerer(kb, api_key=key or None, workspace_id=workspace)
    except Exception:
        # Answering without Claude when the SDK is missing or the client can't be created
        return None


class DupacoKnowledge:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.llm = _claude(self.kb)
        # Caching answers by (normalized question, as_of), so asking twice gives the same answer
        self._answers = {}

    def ingest(self, snapshot: Snapshot):
        self.kb.add_snapshot(snapshot)
        self._answers.clear()

    def answer(self, question, as_of=None):
        # Using the date the question names ("On May 21, 2026, what were ...") when there's no as_of
        as_of = as_of or question_date(self.kb, question)
        key = (" ".join(question.lower().split()), as_of)
        if key not in self._answers:
            self._answers[key] = self._answer(question, as_of)
        return self._answers[key]

    def _answer(self, question, as_of):
        try:
            if self.llm:
                reply = self.llm.answer(question, as_of)
            else:
                reply = fallback_answer(self.kb, question, as_of)
        except Exception:
            # Declining instead of crashing, since declining is a valid answer
            reply = {"text": REFUSAL, "sources": [], "confident": False, "note": None}
        snapshot_id = self.kb.snapshots[self.kb.snap_for(as_of)].snapshot_id if self.kb.snapshots else None
        sources = [Source(url=url, snapshot_id=snapshot_id) for url in reply["sources"]]
        return Answer(text=reply["text"], sources=sources, confident=reply["confident"], conflict_note=reply["note"])

    def conflicts(self):
        found = []
        for c in self.kb.conflicts():
            sources = [Source(url=url, snapshot_id=sid, quote=quote) for url, sid, quote in c.sources]
            found.append(Conflict(topic=c.topic, values=c.values, resolution=c.resolution, reason=c.reason,
                                  sources=sources))
        return found


def build():
    return DupacoKnowledge()
