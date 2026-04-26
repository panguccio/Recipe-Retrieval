import numpy as np
import pickle
from src.core.models import Term
from scipy.sparse import csr_array
from src.utils.tokenize import tokenize

class SearchEngine:
    def __init__(self, index_path, doc_vect_path, weights=None):
        print("Caricamento Search Engine...")
        with open(index_path, "rb") as f:
            self.idx = pickle.load(f)
        with open(doc_vect_path, "rb") as f:
            self.doc_vect = pickle.load(f)
        
        self.vocab = self.idx.vocab
        # Pesi di default se non vengono passati
        self.weights = weights or {"title": 0.7, "instructions": 0.1, "ingredients": 0.2}
        
        # Warmup perchè python è pigro :)
        salamuccio = self.query_to_vector("warmup").dot(self.doc_vect.T)

    def query_to_vector(self, query_text):
        vector = np.zeros(len(self.vocab))
        for token in tokenize(query_text):
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
        if isinstance(query, str):
            vec_q = self.query_to_vector(query)
        else:
            vec_q = query 

        # Calcolo similarità del coseno
        similarities = vec_q.dot(self.doc_vect.T).toarray().flatten()
        top_n_indices = np.argsort(similarities)[::-1][:n]
        return top_n_indices, similarities

    def apply_relevance_feedback(self, vec_q, relevant_ids, all_top_ids):
        relevant_vecs = self.doc_vect[relevant_ids]
        non_relevant_ids = [idx for idx in all_top_ids if idx not in relevant_ids]
        non_relevant_vecs = self.doc_vect[non_relevant_ids]
        return rocchio_algorithm(vec_q, relevant_vecs, non_relevant_vecs)
    
    def set_weights(self, new_weights):
        """Cambia i pesi senza ricaricare l'indice"""
        self.weights = new_weights
        
        
def rocchio_algorithm(query_vec, relevant_vecs, non_relevant_vecs, alpha=1.0, beta=0.75, gamma=0.15):
    """
    Applies the Rocchio algorithm to refine a query vector using relevance feedback.

    Pulls the query toward relevant documents and away from non-relevant ones.
    Alpha, beta, gamma control the weight of the original query, relevant centroid,
    and non-relevant centroid respectively.
    """
    relevant_centroid = (
        sum(relevant_vecs) / relevant_vecs.shape[0]
        if relevant_vecs.shape[0] > 0 else 0
    )
    non_relevant_centroid = (
        sum(non_relevant_vecs) / non_relevant_vecs.shape[0]
        if non_relevant_vecs.shape[0] > 0 else 0
    )

    return (alpha * query_vec) + (beta * relevant_centroid) - (gamma * non_relevant_centroid)