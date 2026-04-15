import json
import re
from nltk.stem import WordNetLemmatizer as wnl

class Recipe: 
    def __init__(self, title, ingredients, instructions):
        self.title = title
        self.ingredients = ingredients
        self.instructions = instructions
    def __repr__(self):
        return self.title


def create_recipe_corpus(filename):
    file = open(filename, "r")
    data = json.load(file)
    recipe_corpus = []
    for index in data:
        recipe_json = data[index]
        if recipe_json and recipe_json.get('ingredients') and recipe_json.get('instructions') and recipe_json.get('title'):
            recipe_corpus.append(Recipe(recipe_json["title"], recipe_json["ingredients"], recipe_json["instructions"]))
    file.close()
    return recipe_corpus

corpus = create_recipe_corpus("recipes/recipes.json")
for i in range(len(corpus)):
    print(f"{i} -> {corpus[i]}")

def normalize(text):
    norm_text = re.sub(r'-', ' ', text)
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text)
    norm_text = norm_text.lower()
    word_list = [wnl().lemmatize(word) for word in norm_text.split()]
    return word_list

def tokenize(recipe):
    title_list = normalize(recipe.title)
    instruction_list = normalize(recipe.instructions)
    return title_list

print(tokenize(corpus[0]))