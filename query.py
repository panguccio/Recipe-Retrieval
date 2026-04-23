import numpy as np
import pickle
import heapq
from core import Term, tokenize, cosine_similarity
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

    while True:
        query = input("\n Enter a query (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break

        vec_q = query_to_vector(query, idx, vocab)

        list_cos = []
        K = 5

        for i, vect in enumerate(doc_vect):
            cos_sim = cosine_similarity(vec_q, vect)

            if len(list_cos) < K:
                heapq.heappush(list_cos, (cos_sim, i))
            else:
                if cos_sim > list_cos[0][0]:
                    heapq.heappop(list_cos)
                    heapq.heappush(list_cos, (cos_sim, i))

    
        list_cos = sorted(list_cos, key=lambda x: x[0], reverse=True)
        print("\nTop 5 most similar documents:")
        for cos_sim, doc_id in list_cos:
            print(f"Document ID: {doc_id}, Cosine Similarity: {cos_sim}")
            print(f"Recipe: {corpus[doc_id]}")

        

