import re
from math import log
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wnl
from sortedcontainers import SortedDict

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


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

    def __gt__(self, other):
        return self.word > other.word

    def __lt__(self, other):
        return self.word < other.word


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

    def add_occurrence(self, doc_id):
        if doc_id not in self._map:
            self._map[doc_id] = Posting(doc_id)
        self._map[doc_id].add_occurrence()

    @property
    def list(self):
        return list(self._map.values())

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map.values())


class InvertedIndex:
    def __init__(self, corpus):
        self.index = SortedDict()
        self.vocab = {}
        self.populate_index(corpus)

    def populate_index(self, corpus):
        total_docs = len(corpus)
        for i, recipe in enumerate(corpus, start=1):
            if i % 1000 == 0:
                print(f"Progress: {i}/{total_docs}")

            for zone in ["title", "instructions", "ingredients"]:
                if zone == "ingredients":
                    text = "".join(recipe.ingredients)
                else:
                    text = getattr(recipe, zone)
                for token in tokenize(text):
                    term = Term(token, zone)
                    if term not in self.index:
                        self.index[term] = PostingList()
                    self.index[term].add_occurrence(recipe.id)
        for term in self.index:
            posting_list = self.index[term]
            term.update_idf(log(len(corpus) / len(posting_list)))
        self.vocab = {term: i for i, term in enumerate(self.index.keys())}

    def __repr__(self):
        return f"{[str(term) + '->' + str(self.index[term]) for term in self.index]}"

    def __len__(self):
        return len(self.index)


def to_wnl_pos(treebank_tag):
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
    norm_text = re.sub(r'\'s', '', norm_text)
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text).lower()

    words_pos = pos_tag([w for w in norm_text.split() if w not in stop_words])
    lemmatizer = wnl()
    return [lemmatizer.lemmatize(w, to_wnl_pos(p)) for w, p in words_pos]
