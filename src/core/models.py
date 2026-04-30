from math import log
from scipy.sparse.linalg import norm

class Term:
    """A single indexed token, scoped to a zone (title, ingredients, instructions)."""

    def __init__(self, word, position):
        self.word = word
        self.position = position  # zone name, e.g. "title"
        self.idf = 0

    def update_idf(self, n_docs, doc_frequency):
        self.idf = log(n_docs / doc_frequency)

    def __eq__(self, other):
        return self.word == other.word and self.position == other.position

    def __hash__(self):
        return hash(self.word + self.position)

    def __repr__(self):
        return f"{self.word}.{self.position}"

    # Ordering is word-only so SortedDict keeps terms alphabetically
    def __gt__(self, other):
        return self.word > other.word

    def __lt__(self, other):
        return self.word < other.word


class Posting:
    """Tracks how many times a term appears in a single document."""

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
    """Collection of Postings for a term, keyed by doc_id."""

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


