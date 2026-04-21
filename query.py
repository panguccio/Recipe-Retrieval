import numpy as np
import pickle
from core import Term, tokenize
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

    vocab = idx.vocab

    while True:
        query = input("\n Enter a query (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
        vec = query_to_vector(query, idx, vocab)
        print(vec)

        