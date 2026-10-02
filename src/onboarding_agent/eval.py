"""Retrieval eval: for each golden question, does search surface the expected handbook doc?"""
import sys

import yaml
from openai import OpenAI

from .config import PROJECT_ROOT, load_settings, require_api_key
from .embeddings import openai_embedder
from .retriever import Retriever

PASS_THRESHOLD = 0.8


def evaluate(retriever: Retriever, cases: list[dict], k: int = 4) -> tuple[float, list[dict]]:
    rows = []
    for case in cases:
        found = [hit.chunk.doc_id for hit in retriever.search(case["question"], k=k)]
        rows.append({"question": case["question"], "expected": case["expect_doc"], "found": found,
                     "hit": case["expect_doc"] in found})
    rate = sum(row["hit"] for row in rows) / len(rows) if rows else 0.0
    return rate, rows


def main() -> None:
    settings = load_settings()
    client = OpenAI(api_key=require_api_key(settings))
    retriever = Retriever.load(settings.index_dir, openai_embedder(client, settings.embed_model))
    cases = yaml.safe_load((PROJECT_ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    rate, rows = evaluate(retriever, cases)
    for row in rows:
        print(f"[{'PASS' if row['hit'] else 'FAIL'}] {row['question']}")
        if not row["hit"]:
            print(f"        expected {row['expected']}, got {row['found']}")
    print(f"\nHit rate @4: {rate:.0%} ({sum(r['hit'] for r in rows)}/{len(rows)})")
    sys.exit(0 if rate >= PASS_THRESHOLD else 1)
