from sortedcontainers import SortedDict
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm
from core import tokenize, Term, PostingList

class InvertedIndex:
    def __init__(self, corpus):
        self.zones = ["title", "instructions", "ingredients"]
        self.index = SortedDict()
        self.vocab = {}
        self.spell = SymSpell(max_dictionary_edit_distance=3)
        self.populate_index(corpus)


    def populate_index(self, corpus):
        for doc_id in tqdm(corpus, desc="Indexing", unit="doc"):
            # for each term add 3 terms in the index, one for each zone
            for zone in self.zones:
                text = corpus[doc_id][zone]
                for token in tokenize(text):
                    term = Term(token, zone)
                    if term not in self.index:
                        self.index[term] = PostingList()
                    self.index[term].add_occurrence(doc_id)

        # calculate the idf for each term at the end
        for position, term in enumerate(self.index):
            posting_list = self.index[term]
            term.update_idf(len(corpus), len(posting_list))
            # update vocab: to easily obtain the position in the index of a term
            self.vocab[term] = position
            self.spell.create_dictionary_entry(term.word, count=1)

    def correct(self, word):
        try:
            suggestions = self.spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            return suggestions[0].term
        except:
            return ""

    def get_postings(self, word, zone=None):
        if zone:
            return self.index[Term(word, zone)].list
        postings = {}
        for zone in self.zones:
            postings[zone] = self.index[Term(word, zone)].list
        return postings

    def debug_print(self, limit = 10):
        return {str(term): str(self.index[term]) for term in self.index.keys()[:limit]}

    def __repr__(self) -> str:
        return (
            f"InvertedIndex("
            f"terms={len(self.index)}, "
            f"vocab_size={len(self.vocab)}, "
            f"zones={self.zones})"
        )

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.index.items())

    def __contains__(self, term):
        return term in self.index