import numpy as np
import pickle
from core import Term, tokenize, cosine_similarity
from corpus_parser import load_corpus
from scipy.sparse import csr_array


def query_to_vector(query, index, vocab):
    vector = np.zeros(len(vocab))

    for token in tokenize(query):
        for zone in ["title", "instructions", "ingredients"]:
            term = Term(token, zone)
            if term in index:
                vector[vocab[term]] += 1
                print(term)
    norma = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma
    return csr_array(vector)

if __name__ == "__main__":

    with open("inverted_index.pkl", "rb") as file:
        idx = pickle.load(file)

    with open("doc_vectors.pkl", "rb") as file:
        doc_vect = pickle.load(file)
    
    vocab = idx.vocab

    corpus = load_corpus("recipes/recipes.json")

    while True:
        query = input("\n Enter a query (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break

        vec_q = query_to_vector(query, idx, vocab)
        
        list_cos_sim = {}
        
        for i, vect in enumerate(doc_vect):
            cos_sim = cosine_similarity(vec_q, vect)
            list_cos_sim[i] = cos_sim
        
        list_cos_sim = sorted(list_cos_sim.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 most similar documents:")
        for doc_id, sim in list_cos_sim[:5]:
            print(f"Document ID: {doc_id}, Cosine Similarity: {sim}")
            print(f"Recipe: {corpus[f"{doc_id}"]}")
        
        