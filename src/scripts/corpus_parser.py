import json
import yaml


def load_corpus(path):
    """Loads and returns a JSON file as a Python dict."""
    with open(path, "r") as source_file:
        return json.load(source_file)


def create_recipe_corpus(source_path, destination_path, update=False):
    """
    Converts a raw recipe dataset into a clean indexed corpus.

    Skips recipes with missing title, ingredients, or instructions.
    If update=True, merges new recipes into an existing corpus file
    instead of rebuilding from scratch.
    """
    with open(source_path, "r") as source_file:
        data = json.load(source_file)

    # If updating, start from the existing corpus so previous entries are preserved
    if update:
        with open(destination_path, "r") as corpus_file:
            recipe_corpus = json.load(corpus_file)
    else:
        recipe_corpus = {}

    doc_id = 0
    for index in data:
        recipe_json = data[index]

        # Skip incomplete records — all three fields are required for indexing
        if not (recipe_json
                and recipe_json.get('ingredients')
                and recipe_json.get('instructions')
                and recipe_json.get('title')):
            continue

        recipe_corpus[doc_id] = {
            'title':        recipe_json.get('title'),
            'ingredients':  "\n".join(recipe_json.get('ingredients')),
            'instructions': recipe_json.get('instructions')
        }
        doc_id += 1

    with open(destination_path, "w") as destination_file:
        json.dump(recipe_corpus, destination_file, indent=4)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Pre-processing the corpus...")
    create_recipe_corpus(config["paths"]["raw_corpus"], config["paths"]["corpus"], update=False)
    print(f"Pre-processing completed. Corpus available at {config["paths"]["corpus"]}")