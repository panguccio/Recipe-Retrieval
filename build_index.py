import json
import pickle
from core import Recipe, InvertedIndex, tokenize
from scipy.sparse import csr_array

def create_recipe_corpus(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    
    recipe_corpus = []
    doc_id = 0
    for index in data:
        recipe_json = data[index]
        if recipe_json and recipe_json.get('ingredients') and recipe_json.get('instructions') and recipe_json.get('title'):
            recipe_corpus.append(Recipe(
                doc_id, 
                recipe_json["title"], 
                recipe_json["ingredients"], 
                recipe_json["instructions"]
            ))
            doc_id += 1
    return recipe_corpus

def build_doc_vectors(inverted_index, corpus):

    number_of_terms = len(inverted_index.index)
    number_of_docs = len(corpus)

    term_to_idx = {term: i for i, term in enumerate(inverted_index.index.keys())}

    # for the doc vectors, we generate a sparse array to avoid wasting space with zeros entries
    # this data structure instead of saving a #docs x #terms matrix, saves the value of tf-idf and the corresponding coordinates
    rows, cols, data = [], [], []

    for term, postings in inverted_index.index.items():
        for posting in postings:
            rows.append(posting.doc_id)
            cols.append(term_to_idx[term])
            data.append(posting.tf * term.idf)

    return csr_array((data, (rows, cols)), shape=(number_of_docs, number_of_terms))

if __name__ == "__main__":

    print("Loading corpus...")
    corpus = create_recipe_corpus("recipes/recipes.json")

    print("Building Inverted Index...")
    idx = InvertedIndex(corpus)

    print("Building Document Vectors...")
    doc_vectors = build_doc_vectors(idx, corpus)

    # saves the inverted index and the document vectors in a pickle file, to use them in testing
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(idx, f)
    print("Inverted Index saved in 'inverted_index.pkl'")

    with open("doc_vectors.pkl", "wb") as f:
        pickle.dump(doc_vectors, f)
    print("Document Vectors saved in 'doc_vectors.pkl'")