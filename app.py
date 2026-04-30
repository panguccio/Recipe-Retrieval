from flask import Flask, render_template, request, jsonify

import yaml
from src.utils.corpus_parser import load_corpus
from src.search.engine import SearchEngine
from src.search.query import print_results


app = Flask(__name__)


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

engine = SearchEngine(
    config['paths']['index_bin'], 
    config['paths']['vectors_bin'],
    weights=config['settings']['default_weights']
)
corpus = load_corpus(config['corpus']['clean'])

top_n_indices = None
vec_q = None

@app.route('/', methods=['GET', 'POST'])
def home():
    global top_n_indices, vec_q
    
    risultati_per_html = []
    query = None

    if request.method == 'POST':
        query = request.form.get('search')
        if query:
            vec_q = engine.query_to_vector(query)
            top_n_indices, similarities = engine.search(vec_q, n=10)

            for doc_id in top_n_indices:
                ricetta = corpus.get(str(doc_id), {})
                risultati_per_html.append({
                    'id': doc_id,
                    'titolo': ricetta.get('title', 'Titolo non trovato'),
                    'score': round(float(similarities[doc_id]), 4),
                    'ingredienti': ricetta.get('ingredients', []),
                    'procedimento': ricetta.get('instructions', [])
                })
    return render_template('index.html', risultati=risultati_per_html, query=query)
        
@app.route('/submit-feedback', methods=['POST'])
def handle_feedback():
    global top_n_indices, vec_q
    risultati_per_html = []
    
    id_string = request.form.get('feedback_ids', '')


    relevant_input = id_string.strip().split()
    relevant_doc_ids = [int(d) for d in relevant_input]

    vec_q = engine.apply_relevance_feedback(vec_q, relevant_doc_ids, top_n_indices)
    top_n_indices, similarities = engine.search(vec_q, n=10)

    for doc_id in top_n_indices:
        ricetta = corpus.get(str(doc_id), {})
        risultati_per_html.append({
            'id': doc_id,
            'titolo': ricetta.get('title', 'Titolo non trovato'),
            'score': round(float(similarities[doc_id]), 4),
            'ingredienti': ricetta.get('ingredients', []),
            'procedimento': ricetta.get('instructions', [])
        })
    return render_template('index.html', risultati=risultati_per_html)
    
    

if __name__ == '__main__':
    app.run(debug=True)