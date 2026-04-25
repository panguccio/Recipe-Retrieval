import json
import time
import numpy as np
import sys
import os

# Gestione percorsi per importare search.py dalla cartella superiore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiamo la classe dal nuovo file search.py
from search import SearchEngine

def benchmark():
    # 1. INIZIALIZZAZIONE ENGINE
    # La classe carica già tutto (index, vectors, vocab) al suo interno
    try:
        # Assicurati che i percorsi siano corretti rispetto a dove lanci lo script
        engine = SearchEngine(
            index_path="benchmark/inverted_index.pkl", 
            doc_vect="benchmark/doc_vectors.pkl"
        )
        
        with open("benchmark/evaluation_set.json", "r") as f:
            test_cases = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Errore caricamento file: {e}")
        print("💡 Suggerimento: lancia lo script dalla root del progetto.")
        return

    results_stats = []
    
    print(f"\n{'Query':<25} | {'P@R':<6} | {'MRR':<6} | {'Time':<8}")
    print("-" * 60)

    for case in test_cases:
        query = case["query"].strip()
        # Convertiamo gli ID attesi in stringhe
        relevant_ids = set(str(rid) for rid in case["relevant_ids"]) 
        
        # 2. ESECUZIONE RICERCA
        start_time = time.time()
        
        # Usiamo il metodo della classe. 
        # Nota: usiamo n=5 o n=len(relevant_ids) a seconda di cosa vuoi testare
        retrieved_indices, similarities = engine.search(query, n=5)
        
        elapsed = time.time() - start_time
        
        # Convertiamo i risultati in stringhe per il confronto
        retrieved_ids = [str(rid) for rid in retrieved_indices]
        
        # 3. CALCOLO PRECISION (Basata sui rilevanti attesi)
        hits = len(set(retrieved_ids) & relevant_ids)
        # Usiamo il numero di rilevanti attesi come divisore (R-Precision) 
        # così se ne aspetti 3 e ne trovi 3, fai 1.00
        p_score = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0
        
        # 4. CALCOLO MRR (Mean Reciprocal Rank)
        rr = 0
        for rank, res_id in enumerate(retrieved_ids, 1):
            if res_id in relevant_ids:
                rr = 1 / rank
                break
        
        results_stats.append({
            "p_score": p_score,
            "rr": rr,
            "time": elapsed
        })
        
        print(f"{query[:25]:<25} | {p_score:<6.2f} | {rr:<6.2f} | {elapsed:<8.4f}s")

    # 5. MEDIE FINALI
    avg_p = np.mean([s["p_score"] for s in results_stats])
    avg_mrr = np.mean([s["rr"] for s in results_stats])
    avg_time = np.mean([s["time"] for s in results_stats])

    print("-" * 60)
    print(f"{'MEDIA FINALE':<25} | {avg_p:<6.2f} | {avg_mrr:<6.2f} | {avg_time:<8.4f}s")

if __name__ == "__main__":
    benchmark()