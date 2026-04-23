import re
from math import log
from scipy.sparse.linalg import norm
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wnl
from nltk.stem import PorterStemmer

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

class Term:
    def __init__(self, word, position):
        self.word = word
        self.position = position
        self.idf = 0

    def update_idf(self, n_docs, doc_frequency):
        self.idf = log(n_docs / doc_frequency)

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

lemmatizer = wnl()
stemmer = PorterStemmer()

def tokenize(text, stemming=False, simple=False):
    norm_text = re.sub(r'-', ' ', text)
    norm_text = re.sub(r'\'s', '', norm_text)
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text).lower()

    if stemming:
        return [stemmer.stem(w) for w in norm_text.split() if w not in stop_words]

    if simple:
        return [lemmatizer.lemmatize(w, "v") for w in norm_text.split() if w not in stop_words]

    words_pos = pos_tag([w for w in norm_text.split() if w not in stop_words])
    lemmatizer = wnl()
    return [lemmatizer.lemmatize(w, to_wnl_pos(p)) for w, p in words_pos]


def cosine_similarity(vec1, vec2):
    vec2 = vec2 / norm(vec2) if norm(vec2) > 0 else vec2   
    
    norm1 = norm(vec1) 
    norm2 = norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return (vec1.dot(vec2)) / (norm1 * norm2)

