import json
import pickle
from math import log
from core import Recipe, InvertedIndex, tokenize

def create_recipe_corpus(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    
    recipe_corpus = []
    doc_id = 1
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

def build_forward_index(inverted_index):
    # { doc_id: { term_object: tfidf_weight } }
    forward_index = {}
    total_terms = len(inverted_index.index)
        
    for i, (term, posting_list) in enumerate(inverted_index.index.items()):
        idf = term.idf
        for doc_id, posting in posting_list._map.items():
            if doc_id not in forward_index:
                forward_index[doc_id] = {}
            
            # TF-IDF
            forward_index[doc_id][term] = posting.tf * idf
        
        if i % 10000 == 0:
            print(f"Progress: {i}/{total_terms}")
            
    return forward_index

if __name__ == "__main__":
    # 1. Creazione del Corpus
    print("Loading corpus...")
    corpus = create_recipe_corpus("recipes/recipes.json")
    
    # 2. Costruzione e salvataggio Inverted Index
    print("Building Inverted Index...")
    idx = InvertedIndex(corpus)
    
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(idx, f)
    print("Inverted Index saved in 'inverted_index.pkl'")

    # 3. Costruzione e salvataggio Forward Index
    print("Generating Foward Index...")
    recipe_vectors = build_forward_index(idx)
    
    with open("forward_index.pkl", "wb") as f:
        pickle.dump(recipe_vectors, f)
    print("Forward Index saved in 'forward_index.pkl'")