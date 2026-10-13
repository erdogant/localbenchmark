"""
Company documents for QA suites.

Put every document in one folder (default: company_docs/). Files are used in
name order, so --limit 40 uses the first 40. Use --ids to choose exact files.
Suite items name the documents that contain their answer in "context", using
file names relative to the folder.

Token counts are estimates (words x 1.35). The real count is the prompt_tokens
value in the results after a run.

  python docs.py estimate company_docs/            # size and window fit
  python docs.py estimate company_docs/ --limit 40
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

DOC_EXTENSIONS = {".txt", ".md"}
TOKENS_PER_WORD = 1.35
ANSWER_HEADROOM = 4096 + 1000       # answer budget plus question and template text
WINDOWS = [50_000, 100_000, 120_000, 150_000, 180_000]


@dataclass
class Doc:
    id: str            # file name, used in suite items
    text: str
    words: int
    est_tokens: int


def estimate_tokens(text):
    return round(len(text.split()) * TOKENS_PER_WORD)


def load_docs(folder="company_docs", limit=None, ids=None):
    """Load documents in name order. `ids` selects exact files (kept in the order given)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"document folder not found: {folder.resolve()}")
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no {'/'.join(sorted(DOC_EXTENSIONS))} files in {folder.resolve()}")

    docs = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        docs.append(Doc(id=p.name, text=text, words=len(text.split()), est_tokens=estimate_tokens(text)))

    if ids is not None:
        by_id = {d.id: d for d in docs}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"documents not in {folder}: {missing}")
        docs = [by_id[i] for i in ids]
    elif limit is not None:
        docs = docs[:limit]
    return docs


def build_context(docs):
    """The text that goes into the prompt, one labelled block per document."""
    return "\n\n".join(f"### Document: {d.id}\n{d.text}" for d in docs)


def fit_report(docs):
    total = sum(d.est_tokens for d in docs)
    lines = [f"{len(docs)} documents, {sum(d.words for d in docs):,} words, "
             f"about {total:,} tokens (estimate)"]
    for w in WINDOWS:
        ok = total + ANSWER_HEADROOM <= w
        lines.append(f"  {w // 1000:>4}k window: {'fits' if ok else 'does NOT fit'}"
                     f" (headroom {w - total - ANSWER_HEADROOM:,} tokens)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Inspect the company document folder.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    est = sub.add_parser("estimate", help="show document count, size and window fit")
    est.add_argument("folder", nargs="?", default="company_docs")
    est.add_argument("--limit", type=int, default=None, help="only the first N documents")
    est.add_argument("--ids", nargs="+", default=None, help="only these file names")
    args = ap.parse_args()
    docs = load_docs(args.folder, limit=args.limit, ids=args.ids)
    print(fit_report(docs))


if __name__ == "__main__":
    main()
