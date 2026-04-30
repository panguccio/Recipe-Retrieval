
import os
import pickle
import yaml
from src.core.index import InvertedIndex
from scipy.sparse import csr_array
from sklearn.preprocessing import normalize
from src.utils.corpus_parser import load_corpus


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
    return  normalize(vectors, norm='l2', axis=1)

def build_index():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loading corpus...")
    corpus = load_corpus(config['corpus']['clean'])

    print("Building Inverted Index...")
    idx = InvertedIndex(corpus, zones=config['settings']['zones'])

    print("Building Document Vectors...")
    doc_vectors = build_doc_vectors(idx, corpus)

    # saves the inverted index and the document vectors in a pickle file, to use them in testing
    index_path = config['paths']['index_bin']
    with open(index_path, "wb") as f:
        pickle.dump(idx, f)
    print(f"Inverted Index saved in '{index_path}'")

    vector_path = config['paths']['vectors_bin']
    with open(vector_path, "wb") as f:
        pickle.dump(doc_vectors, f)
    print(f"Document Vectors saved in '{vector_path}'")
    
    return idx, doc_vectors

if __name__ == "__main__":
    build_index()