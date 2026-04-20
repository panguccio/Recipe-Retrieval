import pickle
from core import Term, PostingList, Posting
from itertools import islice
with open("forward_index.pkl", "rb") as file:
    recipe_vectors = pickle.load(file)
with open("inverted_index.pkl", "rb") as file:
    idx = pickle.load(file)

print(recipe_vectors[1])

count = 0
for term, posting_list in idx.index.items():
    print(f"Termine: {term} | Ricette: {len(posting_list)}")
    
    count += 1
    if count == 10:
        break