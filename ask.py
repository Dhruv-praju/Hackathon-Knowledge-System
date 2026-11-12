"""Ask custom questions.

One question:   python ask.py "question" [--as-of 2026-06-01] [--submission starter]
Interactive:    python ask.py [--submission facts]     (ingests once, then asks in a loop)
                  at the prompt: type a question, or "asof 2026-06-01" to set the date, "asof" to clear, "quit" to exit
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "harness"))
from run_eval import ensure_snapshots, load_snapshots, load_submission  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("question", nargs="*")
p.add_argument("--as-of", default=None)
p.add_argument("--submission", default="starter")
args = p.parse_args()

root = Path(__file__).parent / "snapshots"
ensure_snapshots(root)
ks = load_submission(Path(__file__).parent / args.submission)
for s in load_snapshots(root):
    ks.ingest(s)


def show(question, as_of):
    """Print the answer, its confidence, sources and any conflict note."""
    a = ks.answer(question, as_of=as_of)
    print(a.text)
    print("\nconfident:", a.confident)
    for s in a.sources:
        print("source:", s.url, s.snapshot_id)
    if a.conflict_note:
        print("conflict:", a.conflict_note)


if args.question:
    show(" ".join(args.question), args.as_of)
else:
    as_of = args.as_of
    print("Ready. Ask a question ('asof YYYY-MM-DD' sets the date, 'quit' exits).")
    while True:
        try:
            line = input(f"\n[as_of={as_of or 'latest'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break
        if line.lower().startswith("asof"):
            as_of = line[4:].strip() or None
            continue
        show(line, as_of)
