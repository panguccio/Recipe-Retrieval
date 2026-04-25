import json
import numpy as np
import sys
import os

# Gestione percorsi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search import SearchEngine


def run_test(engine, test_cases):
    """Esegue i test per una singola configurazione e restituisce le medie."""
    rrs = []
    pscores = []

    for case in test_cases:
        query = case["query"].strip()
        relevant_ids = set(str(rid) for rid in case["relevant_ids"])

        # Esecuzione ricerca (l'engine userà i pesi attualmente settati)
        retrieved_indices, _ = engine.search(query, n=5)
        retrieved_ids = [str(rid) for rid in retrieved_indices]

        # P@R (Precision at Recall)
        hits = len(set(retrieved_ids) & relevant_ids)
        pscores.append(hits / len(relevant_ids)
                       if len(relevant_ids) > 0 else 0)

        # MRR
        rr = 0
        for rank, res_id in enumerate(retrieved_ids, 1):
            if res_id in relevant_ids:
                rr = 1 / rank
                break
        rrs.append(rr)

    return np.mean(pscores), np.mean(rrs)


def benchmark():
    try:
        # 1. CARICAMENTO ASSET FISSI
        engine = SearchEngine(
            index_path="benchmark/inverted_index_test.pkl",
            doc_vect_path="benchmark/doc_vectors_test.pkl"
        )

        with open("benchmark/evaluation_set.json", "r") as f:
            test_cases = json.load(f)

        with open("benchmark/weights_configs.json", "r") as f:
            config_data = json.load(f)

    except FileNotFoundError as e:
        print(f"❌ Errore caricamento file: {e}")
        return

    best_mrr = -1
    best_config = None

    print(f"\n{'CONFIG NAME':<25} | {'AVG P@R':<8} | {'AVG MRR':<8}")
    print("-" * 50)

    # 2. CICLO SU TUTTE LE CONFIGURAZIONI
    for cfg in config_data["configurations"]:
        # Aggiorniamo i pesi nell'engine al volo
        engine.weights = cfg["weights"]

        avg_p, avg_mrr = run_test(engine, test_cases)

        print(f"{cfg['name'][:25]:<25} | {avg_p:<8.2f} | {avg_mrr:<8.2f}")

        # Salviamo la migliore basandoci sull'MRR (metrica principe del ranking)
        if avg_mrr > best_mrr:
            best_mrr = avg_mrr
            best_config = cfg

    # 3. VERDETTO FINALE
    print("-" * 50)
    print(f"\n🏆 VINCITORE: {best_config['name']}")
    print(f"📊 MRR Migliore: {best_mrr:.4f}")
    print(f"⚙️  Pesi Ottimali: {best_config['weights']}")


if __name__ == "__main__":
    benchmark()
