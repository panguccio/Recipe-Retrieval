import numpy as np
import pickle
from core import Term, tokenize, rochio_algorithm
from corpus_parser import load_corpus
from scipy.sparse import csr_array

ZONE_WEIGHTS = {
    "title": 0.7,
    "instructions": 0.1,
    "ingredients": 0.2
}


def query_to_vector(query, index, vocab):
    vector = np.zeros(len(vocab))

    for token in tokenize(query):
        token = index.correct(token)
        for zone in ["title", "instructions", "ingredients"]:
            term = Term(token, zone)
            if term in index:
                vector[vocab[term]] += ZONE_WEIGHTS[zone]
    norma = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma
    return csr_array(vector)


def search_top_n(vec_q, n):
    similarities = vec_q.dot(doc_vect.T).toarray().flatten()
    top_n_indices = np.argsort(similarities)[::-1][:n]

    return top_n_indices, similarities


def print_top_n(top_n_indices, similarities, corpus):
    print("\nTop 5 most similar documents:")
    for doc_id in top_n_indices:
        print(f"Document ID: {doc_id}, Cosine Similarity: {similarities[doc_id]}")
        print(f"Recipe: {corpus[f'{doc_id}']['title']}")

if __name__ == "__main__":

    with open("inverted_index.pkl", "rb") as file:
        idx = pickle.load(file)

    with open("doc_vectors.pkl", "rb") as file:
        doc_vect = pickle.load(file)

    vocab = idx.vocab
    corpus = load_corpus("recipes/recipes.json")

    # Addresses lazy initialization that makes the first query very slow
    dummy_warmup = query_to_vector(
        "test query", idx, vocab).dot(doc_vect.T).toarray()


    while True:
        query = input("\n Enter a query (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break

        vec_q = query_to_vector(query, idx, vocab)
        
        
        for iteration in range(2):
            top_n_indices, similarities = search_top_n(vec_q, 10)
            print_top_n(top_n_indices, similarities, corpus)

            if iteration == 1:
                break
            
            print("To Relevance Feedback, enter the document IDs of relevant documents separated by commas (or press Enter to skip):")
            relevant_input = input().strip()

            if relevant_input:
                relevant_doc_ids = [int(doc_id.strip()) for doc_id in relevant_input.split(",") if doc_id.strip().isdigit()]
            
                relevant_vecs = doc_vect[relevant_doc_ids]
                non_relevant_vecs = doc_vect[[doc_id for doc_id in top_n_indices if doc_id not in relevant_doc_ids]]
            
                vec_q = rochio_algorithm(vec_q, relevant_vecs, non_relevant_vecs)
            
               
        
        