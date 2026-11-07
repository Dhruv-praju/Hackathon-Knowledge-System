"""Ask a custom question: python ask.py "question" [--as-of 2026-06-01] [--submission starter]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "harness"))
from run_eval import ensure_snapshots, load_snapshots, load_submission  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("question", nargs="+")
p.add_argument("--as-of", default=None)
p.add_argument("--submission", default="starter")
args = p.parse_args()

root = Path(__file__).parent / "snapshots"
ensure_snapshots(root)
ks = load_submission(Path(__file__).parent / args.submission)
for s in load_snapshots(root):
    ks.ingest(s)

a = ks.answer(" ".join(args.question), as_of=args.as_of)
print(a.text)
print("\nconfident:", a.confident)
for s in a.sources:
    print("source:", s.url, s.snapshot_id)
if a.conflict_note:
    print("conflict:", a.conflict_note)
