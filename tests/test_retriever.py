import numpy as np
import pytest

from fakes import fake_embed
from onboarding_agent.retriever import IndexNotFoundError, Retriever


def test_search_ranks_relevant_chunk_first(retriever):
    hits = retriever.search("How many days of paid leave do I get?")
    assert hits[0].chunk.doc_id == "02-leave"
    assert hits[0].score > hits[-1].score


def test_search_finds_vpn_section(retriever):
    assert retriever.search("How do I connect to the VPN?")[0].chunk.heading == "VPN"


def test_search_respects_k_and_small_indexes(retriever):
    assert len(retriever.search("vpn", k=2)) == 2
    assert len(retriever.search("vpn", k=50)) == len(retriever.chunks) == 4


def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_doc_catalog_lists_each_doc_once_in_order(retriever):
    assert retriever.doc_catalog() == {
        "01-setup": "Dev Setup",
        "02-leave": "Leave Policy",
        "03-workflow": "Engineering Workflow",
    }


def test_load_without_index_raises_helpful_error(tmp_path):
    with pytest.raises(IndexNotFoundError, match="uv run ingest"):
        Retriever.load(tmp_path / "missing", fake_embed)


def test_mismatched_chunks_and_vectors_rejected(retriever):
    with pytest.raises(ValueError):
        Retriever(retriever.chunks, np.zeros((1, 3)), fake_embed)
