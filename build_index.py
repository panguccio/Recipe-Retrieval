
import pickle
from core import InvertedIndex, tokenize
from scipy.sparse import csr_array
from sklearn.preprocessing import normalize
from corpus_parser import load_corpus


def build_doc_vectors(inverted_index, corpus):

    number_of_terms = len(inverted_index)
    number_of_docs = len(corpus)

    vocab = inverted_index.vocab

    # for the doc vectors, we generate a sparse array to avoid wasting space with zeros entries
    # this data structure instead of saving a #docs x #terms matrix, saves the value of tf-idf and the corresponding coordinates
    rows, cols, data = [], [], []

    for term, postings in inverted_index:
        for posting in postings:
            rows.append(posting.doc_id)
            cols.append(vocab[term])
            data.append(posting.tf * term.idf)

    # generate the sparse matrix
    vectors = csr_array((data, (rows, cols)), shape=(number_of_docs, number_of_terms))
    # normalize the document vectors with euclidian norm
    return  normalize(vectors, norm='l2', axis=1)

if __name__ == "__main__":

    print("Loading corpus...")
    corpus = load_corpus("recipes/recipes.json")

    print("Building Inverted Index...")
    idx = InvertedIndex(corpus)
    print(idx)

    print("Building Document Vectors...")
    doc_vectors = build_doc_vectors(idx, corpus)

    # saves the inverted index and the document vectors in a pickle file, to use them in testing
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(idx, f)
    print("Inverted Index saved in 'inverted_index.pkl'")

    with open("doc_vectors.pkl", "wb") as f:
        pickle.dump(doc_vectors, f)
    print("Document Vectors saved in 'doc_vectors.pkl'")