import json
import yaml
import requests, zipfile, io, os
import shutil
from pathlib import Path

URL = "https://eightportions.com/recipes_raw.zip"
RAW_JSON = "recipes_raw_nosource_fn.json"

def download_corpus(url, dir, raw_path):
    
    print("Downloading the file...")
    
    raw_dir = os.path.join(dir, "raw")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(raw_dir)
    for file in os.listdir(raw_dir):
        filename = os.fsdecode(file)
        if filename != RAW_JSON:
            os.remove(os.path.join(raw_dir, filename))
    
    source_file = os.path.join(raw_dir, RAW_JSON)
    
    shutil.move(str(source_file), str(raw_path))
    os.rmdir(raw_dir)
    
    print(f"Download completed: saved in {str(raw_path)}")

def load_corpus(path):
    """Loads and returns a JSON file as a Python dict."""
    with open(path, "r") as source_file:
        data = json.load(source_file)
    return data


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


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    DIR = config["corpus"]["dir"]
    RAW_PATH = config["corpus"]["raw"]
    CLEAN_PATH = config["corpus"]["clean"]
    
    # download_corpus(URL, DIR, RAW_PATH)
    
    print("Pre-processing the corpus...")
    create_recipe_corpus(RAW_PATH, CLEAN_PATH, update=False)
    print(f"Pre-processing completed. Corpus available at {CLEAN_PATH}")


if __name__ == "__main__":
    
    main()