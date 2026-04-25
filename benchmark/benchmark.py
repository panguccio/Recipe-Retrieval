import json
import numpy as np
import sys
import os

# Gestione percorsi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search import SearchEngine


def run_test(engine, test_cases):
    """Esegue i test calcolando Precision, Recall e MRR."""
    precisions = []
    recalls = []
    rrs = []

    for case in test_cases:
        query = case["query"].strip()
        relevant_ids = set(str(rid) for rid in case["relevant_ids"])

        # Esecuzione ricerca (Top 5)
        retrieved_indices, _ = engine.search(query, n=5)
        retrieved_ids = [str(rid) for rid in retrieved_indices]

        # Calcolo HITS (quanti ne abbiamo indovinati)
        hits = len(set(retrieved_ids) & relevant_ids)

        # 1. PRECISION @ 5 = Hits / Risultati restituiti (5)
        # "Di quelli che ti ho dato, quanti erano giusti?"
        precisions.append(hits / len(retrieved_ids)
                          if len(retrieved_ids) > 0 else 0)

        # 2. RECALL @ 5 = Hits / Totale rilevanti esistenti
        # "Di tutti quelli giusti che esistono, quanti ne hai presi?"
        recalls.append(hits / len(relevant_ids)
                       if len(relevant_ids) > 0 else 0)

        # 3. MRR
        rr = 0
        for rank, res_id in enumerate(retrieved_ids, 1):
            if res_id in relevant_ids:
                rr = 1 / rank
                break
        rrs.append(rr)

    return np.mean(precisions), np.mean(recalls), np.mean(rrs)


def benchmark():
    try:
        engine = SearchEngine(
            index_path="benchmark/inverted_index_test.pkl",
            doc_vect_path="benchmark/doc_vectors_test.pkl"
        )
        with open("benchmark/evaluation_set.json", "r") as f:
            test_cases = json.load(f)
        with open("benchmark/weights_configs.json", "r") as f:
            config_data = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Errore: {e}")
        return

    best_mrr = -1
    best_composite = 0
    best_config = None

    print(f"\n{'CONFIG NAME':<25} | {'PREC':<6} | {'REC':<6} | {'MRR':<6}")
    print("-" * 65)

    for cfg in config_data["configurations"]:
        engine.weights = cfg["weights"]
        avg_p, avg_r, avg_mrr = run_test(engine, test_cases)

        print(
            f"{cfg['name'][:25]:<25} | {avg_p:<6.2f} | {avg_r:<6.2f} | {avg_mrr:<6.2f}")

        composite = (avg_p + avg_r + avg_mrr) / 3
        if composite > best_composite:
            best_composite = composite
            best_config = cfg

    print("-" * 65)
    print(f"\n🏆 VINCITORE: {best_config['name']}")
    print(
        f"📊 Risultati: Prec: {avg_p:.2f}, Rec: {avg_r:.2f}, MRR: {best_mrr:.4f}")
    print(f"⚙️  Pesi: {best_config['weights']}")


if __name__ == "__main__":
    benchmark()
