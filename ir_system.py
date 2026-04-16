import json
import re
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer as wnl
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')


class Recipe:
    def __init__(self, id, title, ingredients, instructions):
        self.id = id
        self.title = title
        self.ingredients = ingredients
        self.instructions = instructions

    def __repr__(self):
        return self.title


class Term:
    def __init__(self, word, position):
        self.word = word
        self.position = position

    def __eq__(self, other):
        return (self.word == other.word and self.position == other.position)


class PostingList:
    def __init__(self):
        self.list = []
#we will ignore tf, because we are cooked
    def add_occurence(self, doc_id):
        if doc_id not in self.list:
            self.list.append(doc_id)

class InvertedIndex:
    def __init__(self, corpus):
        self.index = {}
        self.populate_index(corpus)

    def populate_index(self, corpus):
        for recipe in corpus:

            for title_token in tokenize(recipe.title):
                term = Term(title_token, "title")
                if term not in self.index:
                    self.index[term] = PostingList()
                self.index[term].add_occurence(recipe.id)

def create_recipe_corpus(filename):
    file = open(filename, "r")
    data = json.load(file)
    recipe_corpus = []
    id = 1
    for index in data:
        recipe_json = data[index]
        if recipe_json and recipe_json.get('ingredients') and recipe_json.get('instructions') and recipe_json.get('title'):
            recipe_corpus.append(Recipe(
                id, recipe_json["title"], recipe_json["ingredients"], recipe_json["instructions"]))
            id += 1
    file.close()
    return recipe_corpus


corpus = create_recipe_corpus("recipes/recipes.json")


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def tokenize(text):
    norm_text = re.sub(r'-', ' ', text)
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text)
    norm_text = norm_text.lower()
    word_list = [wnl().lemmatize(word) for word in norm_text.split()]
    return word_list


tuples = pos_tag(tokenize(corpus[1]))
for word, pos in tuples:
    print(wnl().lemmatize(word, get_wordnet_pos(pos)))
