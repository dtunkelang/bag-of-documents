"""Tests for ESCI evaluation metric computation."""

import pandas as pd


def compute_esci_metrics(query_to_retrieved, esci_df, title_set):
    """Extract the metric computation logic for testing."""
    esci_in_index = esci_df[esci_df["product_title"].isin(title_set)].copy()

    results = {
        "Exact": {"retrieved": 0, "total": 0},
        "Substitute": {"retrieved": 0, "total": 0},
        "Complement": {"retrieved": 0, "total": 0},
        "Irrelevant": {"retrieved": 0, "total": 0},
    }

    for _, row in esci_in_index.iterrows():
        label = row["esci_label"]
        query = row["query"]
        title = row["product_title"]
        results[label]["total"] += 1
        if query in query_to_retrieved and title in set(query_to_retrieved[query]):
            results[label]["retrieved"] += 1

    relevant_retrieved = results["Exact"]["retrieved"] + results["Substitute"]["retrieved"]
    relevant_total = results["Exact"]["total"] + results["Substitute"]["total"]
    irrelevant_retrieved = results["Complement"]["retrieved"] + results["Irrelevant"]["retrieved"]
    total_retrieved = relevant_retrieved + irrelevant_retrieved
    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall = relevant_retrieved / relevant_total if relevant_total > 0 else 0

    return {
        "results": results,
        "precision": precision,
        "recall": recall,
    }


def test_perfect_retrieval():
    """All Exact products retrieved, no junk."""
    esci_df = pd.DataFrame(
        {
            "query": ["laptop", "laptop", "laptop"],
            "product_title": ["Good Laptop", "OK Laptop", "Bad Thing"],
            "esci_label": ["Exact", "Substitute", "Irrelevant"],
        }
    )
    title_set = {"Good Laptop", "OK Laptop", "Bad Thing"}
    query_to_retrieved = {"laptop": ["Good Laptop", "OK Laptop"]}

    m = compute_esci_metrics(query_to_retrieved, esci_df, title_set)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_partial_retrieval():
    """One Exact retrieved, one missed."""
    esci_df = pd.DataFrame(
        {
            "query": ["laptop", "laptop"],
            "product_title": ["Good Laptop", "Also Good Laptop"],
            "esci_label": ["Exact", "Exact"],
        }
    )
    title_set = {"Good Laptop", "Also Good Laptop"}
    query_to_retrieved = {"laptop": ["Good Laptop"]}

    m = compute_esci_metrics(query_to_retrieved, esci_df, title_set)
    assert m["recall"] == 0.5
    assert m["precision"] == 1.0


def test_complement_hurts_precision():
    """Retrieving complements should lower precision."""
    esci_df = pd.DataFrame(
        {
            "query": ["laptop", "laptop"],
            "product_title": ["Good Laptop", "Laptop Case"],
            "esci_label": ["Exact", "Complement"],
        }
    )
    title_set = {"Good Laptop", "Laptop Case"}
    query_to_retrieved = {"laptop": ["Good Laptop", "Laptop Case"]}

    m = compute_esci_metrics(query_to_retrieved, esci_df, title_set)
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0


def test_no_retrieval():
    """Nothing retrieved."""
    esci_df = pd.DataFrame(
        {
            "query": ["laptop"],
            "product_title": ["Good Laptop"],
            "esci_label": ["Exact"],
        }
    )
    title_set = {"Good Laptop"}
    query_to_retrieved = {}

    m = compute_esci_metrics(query_to_retrieved, esci_df, title_set)
    assert m["precision"] == 0
    assert m["recall"] == 0


def test_products_not_in_index_excluded():
    """Products not in title_set should not count."""
    esci_df = pd.DataFrame(
        {
            "query": ["laptop", "laptop"],
            "product_title": ["Good Laptop", "Unknown Product"],
            "esci_label": ["Exact", "Exact"],
        }
    )
    title_set = {"Good Laptop"}  # Unknown Product not in index
    query_to_retrieved = {"laptop": ["Good Laptop"]}

    m = compute_esci_metrics(query_to_retrieved, esci_df, title_set)
    assert m["results"]["Exact"]["total"] == 1  # only 1 in index
    assert m["recall"] == 1.0
