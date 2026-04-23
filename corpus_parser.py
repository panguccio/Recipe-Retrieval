import json

def load_corpus(path):
    with open(path, "r") as source_file:
        data = json.load(source_file)
    return data

def create_recipe_corpus(source_path, destination_path, update=False):
    with open(source_path, "r") as source_file:
        data = json.load(source_file)

    recipe_corpus = {}
    if update:
        with open(destination_path, "r") as corpus_file:
            recipe_corpus = json.load(corpus_file)

    doc_id = 0
    for index in data:
        recipe_json = data[index]
        if recipe_json and recipe_json.get('ingredients') and recipe_json.get('instructions') and recipe_json.get(
                'title'):
            recipe_corpus[doc_id] = {
                'title': recipe_json.get('title'),
                'ingredients': "\n".join(recipe_json.get('ingredients')),
                'instructions': recipe_json.get('instructions')
            }
            doc_id += 1

    with open(destination_path, "w") as destination_file:
        json.dump(recipe_corpus, destination_file, indent=4)





if __name__ == "__main__":
    create_recipe_corpus("recipes/recipes_raw.json", "recipes/recipes.json", update=False)

