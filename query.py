import numpy as np
import pickle
from core import Term, tokenize
from build_index import create_recipe_corpus
from scipy.sparse import csr_array


def query_to_vector(query, index, vocab):
    vector = np.zeros(len(vocab))

    for token in tokenize(query):
        for zone in ["title", "instructions", "ingredients"]:
            token = index.correct(token)
            term = Term(token, zone)
            if term in index:
                vector[vocab[term]] += 1
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
    corpus = create_recipe_corpus("recipes/recipes.json")
    
    #Addresses lazy initialization that makes the first query very slow
    dummy_warmup = query_to_vector("test query", idx, vocab).dot(doc_vect.T).toarray()
    
    while True:
        query = input("\n Enter a query (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break

        vec_q = query_to_vector(query, idx, vocab)

        similarities = vec_q.dot(doc_vect.T).toarray().flatten()

        top_5_indices = np.argsort(similarities)[::-1][:5]

        print("\nTop 5 most similar documents:")
        for doc_id in top_5_indices:

            print(
                f"Document ID: {doc_id}, Cosine Similarity: {similarities[doc_id]}")
            print(f"Recipe: {corpus[doc_id]}")
