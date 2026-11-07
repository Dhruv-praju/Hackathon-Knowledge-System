"""
Starter solution — deliberately naive. It works, and it scores badly on purpose.

It keeps only the LATEST copy of each page and returns the paragraph that shares the most
words with the question. It never notices that pages disagree, ignores `as_of`, and
returns no conflicts. Your job is to do much better.

Keep the `build()` function and the three methods; change everything else.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

WORD = re.compile(r"[a-z0-9$%.]+")


class NaiveKnowledge:
    """Naive keyword-search knowledge base over website snapshots.

    Stores only the latest version of each page and answers questions by
    returning the paragraph with the greatest word overlap with the question.
    """

    def __init__(self):
        """Initialize an empty store mapping url -> (snapshot_id, markdown)."""
        self.pages = {}

    def ingest(self, snapshot: Snapshot) -> None:
        """Ingest one snapshot, keeping only the newest copy of each page.

        Args:
            snapshot: A snapshot of the website. Called once per snapshot,
                oldest first, so later snapshots overwrite earlier pages.
        """
        for p in snapshot.pages:
                self.pages[p.url] = (snapshot.snapshot_id, p.markdown)

    def answer(self, question, as_of=None):
        """Answer a question with the best keyword-matching paragraph.

        Args:
            question: Natural-language question to answer.
            as_of: Optional point in time for the answer. Currently ignored.

        Returns:
            An Answer containing the best-matching paragraph (up to 600 chars)
            and its source, or a non-confident "I don't know." if no paragraph
            shares a word with the question.
        """
        q = set(WORD.findall(question.lower()))
        best, best_score = None, 0
        for url, (sid, md) in self.pages.items():
            for para in md.split("\n\n"):
                s = len(q & set(WORD.findall(para.lower())))
                if s > best_score:
                    best, best_score = (url, sid, para.strip()), s
        if not best:
            return Answer(text="I don't know.", confident=False)
        url, sid, para = best
        return Answer(text=para[:600], sources=[Source(url=url, snapshot_id=sid)])

    def conflicts(self):
        """Return detected conflicts between pages (none; not implemented)."""
        return []


def build():
    """Create and return the knowledge-base instance used by the harness."""
    return NaiveKnowledge()
