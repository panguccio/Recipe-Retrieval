import yaml
from src.utils.corpus_parser import load_corpus
from src.search.engine import SearchEngine

def print_results(top_n_indices, similarities, corpus):
    """Prints the top search results with their similarity scores."""
    print("\n--- Top 10 Results ---")
    for doc_id in top_n_indices:
        title = corpus.get(str(doc_id), {}).get('title')
        ingredients = corpus.get(str(doc_id), {}).get('ingredients')
        instructions = corpus.get(str(doc_id), {}).get('instructions')
        print(f"ID: {doc_id} | Sim: {similarities[doc_id]:.4f} | Recipe: {title}\n\nIngredients:\n{ingredients}\n\nInstructions: {instructions}\n {"-"*20}")

def cli_search():
    # Caricamento configurazione
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Inizializzazione del motore
    engine = SearchEngine(
        config['paths']['index_bin'], 
        config['paths']['vectors_bin'],
        weights=config['settings']['default_weights']
    )
    corpus = load_corpus(config['corpus']['clean'])

    while True:
        query_text = input("\nEnter a query (or 'Enter' to quit): ").strip()
        if query_text == "":
            break

        # Build the initial TF-IDF query vector
        vec_q = engine.query_to_vector(query_text)
        top_n_indices, similarities = engine.search(vec_q, n=10)
        print_results(top_n_indices, similarities, corpus)

        # Relevance feedback loop — repeats until the user presses Enter with no input
        while True:
            print("\nRelevance feedback: enter the IDs of relevant documents, space-separated.")
            print("(Press Enter to skip)")
            relevant_input = input("Relevant IDs: ").strip().split()

            if not relevant_input:
                break

            try:
                relevant_doc_ids = [int(d) for d in relevant_input]
            except ValueError:
                print("Invalid input. Skipping feedback.")
                continue

            # Refine the query vector with Rocchio and re-run the search
            vec_q = engine.apply_relevance_feedback(vec_q, relevant_doc_ids, top_n_indices)
            top_n_indices, similarities = engine.search(vec_q, n=10)
            print_results(top_n_indices, similarities, corpus)

if __name__ == "__main__":
    cli_search()