import json
import numpy as np
import sys
import os

# Allow imports from the parent directory (e.g. the search module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search import SearchEngine


def calculate_ap(retrieved_ids, relevant_ids):
    """
    Computes Average Precision for a single query.

    Args:
        retrieved_ids: ordered list of document IDs returned by the engine
        relevant_ids:  list of ground-truth relevant document IDs

    Returns:
        AP score in [0, 1]; returns 0 if relevant_ids is empty
    """
    actual_relevant_found = 0
    precision_sum = 0

    for i, res_id in enumerate(retrieved_ids, 1):
        if res_id in relevant_ids:
            actual_relevant_found += 1
            # Precision at rank i: how many relevant docs in the top-i results
            precision_sum += (actual_relevant_found / i)

    # Avoid division by zero if no relevant documents exist for this query
    if not relevant_ids:
        return 0

    # Normalise by total number of relevant documents, not just those retrieved
    return precision_sum / len(relevant_ids)


def run_test(engine, test_cases):
    """
    Runs all test queries and returns aggregate retrieval metrics.

    Metrics computed:
      - Precision
      - Recall   
      - MAP

    Returns:
        Tuple (avg_precision, avg_recall, mean_average_precision)
    """
    precisions = []
    recalls = []
    aps = []

    for case in test_cases:
        query = case["query"].strip()
        relevant_ids = [str(rid) for rid in case["relevant_ids"]]

        # Retrieve top-5 results from the search engine
        retrieved_indices, _ = engine.search(query, n=5)
        retrieved_ids = [str(rid) for rid in retrieved_indices]

        # Count how many retrieved documents are actually relevant
        hits = len([rid for rid in retrieved_ids if rid in relevant_ids])

        # Precision: relevant hits / total retrieved
        precisions.append(hits / len(retrieved_ids) if retrieved_ids else 0)

        # Recall: relevant hits / total relevant documents for this query
        recalls.append(hits / len(relevant_ids) if relevant_ids else 0)

        # Average Precision for this query
        aps.append(calculate_ap(retrieved_ids, relevant_ids))

    return np.mean(precisions), np.mean(recalls), np.mean(aps)


def benchmark():
    # Initialise the search engine with pre-built index and document vectors
    engine = SearchEngine(
        index_path="benchmark/inverted_index_test.pkl",
        doc_vect_path="benchmark/doc_vectors_test.pkl"
    )

    # Load the evaluation queries and their ground-truth relevant document IDs
    with open("benchmark/evaluation_set.json", "r") as f:
        test_cases = json.load(f)

    # Load the weight configurations to benchmark
    with open("benchmark/weights_configs.json", "r") as f:
        config_data = json.load(f)

    # Track the best-performing configuration
    best_map = -1.0
    best_config = None
    best_stats = {}

    # Print results table header
    print(f"\n{'CONFIG NAME':<25} | {'P@5':<6} | {'R@5':<6} | {'MAP':<6}")
    print("-" * 55)

    for cfg in config_data["configurations"]:
        # Apply this configuration's field weights to the engine
        engine.set_weights(cfg["weights"])

        # Evaluate across all test queries
        avg_p, avg_r, map_score = run_test(engine, test_cases)

        print(f"{cfg['name'][:25]:<25} | {avg_p:<6.2f} | {avg_r:<6.2f} | {map_score:<6.2f}")

        # Keep track of the config with the highest MAP
        if map_score > best_map:
            best_map = map_score
            best_config = cfg
            best_stats = {"p": avg_p, "r": avg_r, "map": map_score}

    # Print summary of the winning configuration
    print("-" * 55)
    print(f"\nBest weights config: {best_config['name']}")
    print(f"MAP: {best_stats['map']:.4f}")
    print(f"Precision: {best_stats['p']:.2f}, Recall: {best_stats['r']:.2f}")
    print(f"Weights: {best_config['weights']}")


if __name__ == "__main__":
    benchmark()