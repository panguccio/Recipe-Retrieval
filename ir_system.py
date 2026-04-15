import json
import re


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
        ingredients, instructions, title = recipe_json['ingredients'], recipe_json['instructions'], recipe_json['title']
        if recipe_json and ingredients and instructions and title:
            recipe_corpus.append(Recipe(title, ingredients, instructions))
    file.close()
    return recipe_corpus

corpus = create_recipe_corpus("recipes/recipes.json")
for i in range(len(corpus)):
    print(f"{i} -> {corpus[i]}")

def normalize(text):
    norm_text = re.sub(r'-', ' ', text)
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text)
    norm_text = norm_text.lower()
    return list(norm_text.split())

def tokenize(recipe):
    title_list = normalize(recipe.title)
    instruction_list = normalize(recipe.instructions)
    return title_list

