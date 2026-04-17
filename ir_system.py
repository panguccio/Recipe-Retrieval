import json
import re
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer as wnl

# to do: maybe move these to a setup script or something
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

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
        self.idf = 0

    def update_idf(self, idf):
        self.idf = idf

    def __eq__(self, other):
        return self.word == other.word and self.position == other.position

    def __repr__(self):
        return f"{self.word}.{self.position} (idf:{self.idf})"

    def __hash__(self):
        return hash(self.word + self.position)


class Posting:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.tf = 0

    def add_occurrence(self):
        self.tf += 1

    def __eq__(self, other):
        return self.doc_id == other.doc_id

    def __repr__(self):
        return f"(docid:{self.doc_id}, tf:{self.tf})"


class PostingList:
    def __init__(self):
        self._map = {}

    # we will ignore tf, because we are cooked
    def add_occurrence(self, doc_id):
        if doc_id not in self._map:
            self._map[doc_id] = Posting(doc_id)
        self._map[doc_id].add_occurrence()

    @property
    def list(self):
        return list(self._map.values())

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        if len(self.list) == 0:
            return "[]"
        else:
            return f"{self.list}"

class InvertedIndex:
    def __init__(self, corpus):
        self.index = {}
        self.populate_index(corpus)

    def populate_index(self, corpus):
        for recipe in corpus:
            for zone in ["title", "instructions"]:
                text = getattr(recipe, zone)
                print(text)
                for token in tokenize(text):
                    term = Term(token, zone)
                    if term not in self.index:
                        self.index[term] = PostingList()
                    self.index[term].add_occurrence(recipe.id)
        for term in self.index:
            posting_list = self.index[term]
            term.update_idf(len(corpus)/len(posting_list))


    def __repr__(self):
        return f"{[str(term) + '->' + str(self.index[term]) for term in self.index]}"


def create_recipe_corpus(filename):
    file = open(filename, "r")
    data = json.load(file)
    recipe_corpus = []
    doc_id = 1
    for index in data:
        recipe_json = data[index]
        if recipe_json and recipe_json.get('ingredients') and recipe_json.get('instructions') and recipe_json.get(
                'title'):
            recipe_corpus.append(Recipe(
                doc_id, recipe_json["title"], recipe_json["ingredients"], recipe_json["instructions"]))
            doc_id += 1
    file.close()
    return recipe_corpus


def to_wnl_pos(treebank_tag):
    # a map from nltk pos to wordnet pos
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
    # remove -
    norm_text = re.sub(r'-', ' ', text)
    # remove possessive 's
    norm_text = re.sub(r'\'s', '', norm_text)
    # keep just letters and spaces
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text)
    # lower case text
    norm_text = norm_text.lower()
    # tag the words with part of speach
    words_pos = pos_tag(norm_text.split())
    # lemmatize with wordnet
    word_list = [wnl().lemmatize(w, to_wnl_pos(p)) for w, p in words_pos]

    return word_list


corpus = create_recipe_corpus("recipes/test.json")
idx = InvertedIndex(corpus)
print(idx)
