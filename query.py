from corpus_parser import load_corpus
from search import SearchEngine
# Assumiamo che SearchEngine sia definita in search_engine.py o sopra nello stesso file
# from search_engine import SearchEngine 

def print_results(top_n_indices, similarities, corpus):
    print("\n--- Top 10 Risultati ---")
    for doc_id in top_n_indices:
        # Usiamo get() o f'{doc_id}' a seconda di come è strutturato il tuo corpus
        recipe_title = corpus.get(str(doc_id), {}).get('title', 'Titolo non trovato')
        print(f"ID: {doc_id} | Sim: {similarities[doc_id]:.4f} | Ricetta: {recipe_title}")

if __name__ == "__main__":
    # Inizializzazione del motore
    # Nota: i pesi sono gestiti internamente alla classe come abbiamo definito prima
    engine = SearchEngine("inverted_index.pkl", "doc_vectors.pkl")
    corpus = load_corpus("recipes/recipes.json")

    while True:
        query_text = input("\nInserisci una query (o 'q' per uscire): ").strip()
        if query_text.lower() == "q":
            break

        # 1. Generiamo il vettore iniziale della query
        vec_q = engine.query_to_vector(query_text)
        
        # Iterazioni per il Relevance Feedback (Originale + 1 feedback)
        for iteration in range(2):
            # 2. Eseguiamo la ricerca
            top_n_indices, similarities = engine.search(vec_q, n=10)
            print_results(top_n_indices, similarities, corpus)

            if iteration == 1:
                break
            
            print("\nFeedback Rilevanza: inserisci gli ID dei documenti rilevanti separati da virgola")
            print("(Premi Invio per saltare il feedback)")
            
            relevant_input = input("ID rilevanti: ").strip()

            if relevant_input:
                try:
                    # Convertiamo l'input in una lista di interi
                    relevant_doc_ids = [int(d.strip()) for d in relevant_input.split(",") if d.strip().isdigit()]
                    
                    if relevant_doc_ids:
                        # 3. Applichiamo Rocchio per raffinare il vettore vec_q
                        # Passiamo vec_q, i doc scelti dall'utente e i 10 visti (per i non rilevanti)
                        vec_q = engine.apply_relevance_feedback(vec_q, relevant_doc_ids, top_n_indices)
                        print("--- Query raffinata con successo. Riesecuzione... ---")
                    else:
                        break
                except ValueError:
                    print("Input non valido. Salto il feedback.")
                    break
            else:
                # Se l'utente preme invio senza scrivere nulla, usciamo dal ciclo feedback
                break