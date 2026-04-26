from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer as wnl
from nltk import pos_tag
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)

import re
stemmer = PorterStemmer()
lemmatizer = wnl()


def to_wnl_pos(treebank_tag):
    """Converts to the simplified form expected by WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return 'a'  # adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # verb
    elif treebank_tag.startswith('N'):
        return 'n'  # noun
    elif treebank_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun


stop_words = set(stopwords.words('english'))


def tokenize(text, stemming=False, simple=False):
    """Normalises and tokenizes text into lemmatized or stemmed tokens."""

    # Normalize: expand hyphens, strip possessives, remove non-alpha characters
    norm_text = re.sub(r'-', ' ', text)
    norm_text = re.sub(r'\'s', '', norm_text)
    norm_text = re.sub(r'[^a-zA-Z\s]', '', norm_text).lower()

    words = [w for w in norm_text.split() if w not in stop_words]

    if stemming:
        return [stemmer.stem(w) for w in words]
    if simple:
        return [lemmatizer.lemmatize(w, "v") for w in words]

    # Full path: tag each word with its POS then lemmatize accordingly
    words_pos = pos_tag(words)
    return [lemmatizer.lemmatize(w, to_wnl_pos(p)) for w, p in words_pos]