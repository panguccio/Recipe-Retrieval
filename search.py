import numpy as np
import pickle
from core import Term, tokenize
from scipy.sparse import csr_array

class SearchEngine:
    def __init__(self, index_path, doc_vect, weights=None):
        print("Caricamento Search Engine...")
        with open(index_path, "rb") as f:
            self.idx = pickle.load(f)
        with open(doc_vect, "rb") as f:
            self.doc_vect = pickle.load(f)
        
        self.vocab = self.idx.vocab

        #Assegnazione pesi
        self.weights = weights or {"title": 1, "instructions": 0, "ingredients": 1}
        
        # Warmup perchè python è pigro :)
        salamuccio = self.query_to_vector("warmup").dot(self.doc_vect.T)

    def query_to_vector(self, query):
        vector = np.zeros(len(self.vocab))
        for token in tokenize(query):
            token = self.idx.correct(token)
            for zone, weight in self.weights.items():
                term = Term(token, zone)
                if term in self.idx:
                    vector[self.vocab[term]] += weight
        
        norma = np.linalg.norm(vector)
        if norma > 0:
            vector /= norma
        return csr_array(vector)

    def search(self, query, n=10):
        vec_q = self.query_to_vector(query)
        similarities = vec_q.dot(self.doc_vect.T).toarray().flatten()
        top_n_indices = np.argsort(similarities)[::-1][:n]
        return top_n_indices, similarities