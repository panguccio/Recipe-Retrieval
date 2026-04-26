import json
import numpy as np
import sys
import os

# Gestione percorsi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search import SearchEngine

def calculate_ap(retrieved_ids, relevant_ids):
    """
    Calcola la Average Precision
    """
    actual_relevant_found = 0
    precision_sum = 0
    
    for i, res_id in enumerate(retrieved_ids, 1):
        if res_id in relevant_ids:
            actual_relevant_found += 1
            # Precisione calcolata al rango i
            precision_sum += (actual_relevant_found / i)
            
    if not relevant_ids:
        return 0
    
    return precision_sum / len(relevant_ids)

def run_test(engine, test_cases):
    """Esegue i test calcolando Precision@5, Recall@5 e MAP."""
    precisions = []
    recalls = []
    aps = []

    for case in test_cases:
        query = case["query"].strip()
        relevant_ids = [str(rid) for rid in case["relevant_ids"]]
        
        # Esecuzione ricerca
        retrieved_indices, _ = engine.search(query, n=5)
        retrieved_ids = [str(rid) for rid in retrieved_indices]

        # Calcolo HITS per metriche base
        hits = len([rid for rid in retrieved_ids if rid in relevant_ids])
                
        # P@5 e R@5
        precisions.append(hits / len(retrieved_ids) if len(retrieved_ids) > 0 else 0)
        recalls.append(hits / len(relevant_ids) if len(relevant_ids) > 0 else 0)

        # Average Precision
        aps.append(calculate_ap(retrieved_ids, relevant_ids))

    return np.mean(precisions), np.mean(recalls), np.mean(aps)

def benchmark():
    engine = SearchEngine(
        index_path="benchmark/inverted_index_test.pkl",
        doc_vect_path="benchmark/doc_vectors_test.pkl"
    )
    with open("benchmark/evaluation_set.json", "r") as f:
        test_cases = json.load(f)
    with open("benchmark/weights_configs.json", "r") as f:
        config_data = json.load(f)

    best_map = -1.0
    best_config = None
    best_stats = {}

    print(f"\n{'CONFIG NAME':<25} | {'P@5':<6} | {'R@5':<6} | {'MAP':<6}")
    print("-" * 55)

    for cfg in config_data["configurations"]:
        # Aggiorna pesi
        engine.set_weights(cfg["weights"])
        
        # Calcolo metriche
        avg_p, avg_r, map_score = run_test(engine, test_cases)

        print(f"{cfg['name'][:25]:<25} | {avg_p:<6.2f} | {avg_r:<6.2f} | {map_score:<6.2f}")

        # MAP
        if map_score > best_map:
            best_map = map_score
            best_config = cfg
            best_stats = {"p": avg_p, "r": avg_r, "map": map_score}

    print("-" * 55)
    print(f"\n🏆 VINCITORE: {best_config['name']}")
    print(f"📊 MAP (Mean Average Precision): {best_stats['map']:.4f}")
    print(f"📈 Dettagli: Precision@5: {best_stats['p']:.2f}, Recall@5: {best_stats['r']:.2f}")
    print(f"⚙️  Pesi Ottimali: {best_config['weights']}")

if __name__ == "__main__":
    benchmark()