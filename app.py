from flask import Flask, render_template, request

import yaml
from src.utils.corpus_parser import load_corpus
from src.search.engine import SearchEngine


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

def format_results(similarity):
    global top_n_indices, vec_q
    formatted_result = []
    
    for doc_id in top_n_indices:
        ricetta = corpus.get(str(doc_id), {})
        
        raw_ingredients = ricetta.get('ingredients', [])

        lista_ingredienti = [line.strip() for line in raw_ingredients.splitlines() if line.strip()]

        formatted_result.append({
            'id': doc_id,
            'titolo': ricetta.get('title', 'Titolo non trovato'),
            'score': round(float(similarity[doc_id]), 4),
            'ingredienti': lista_ingredienti,
            'procedimento': ricetta.get('instructions', [])
        })

    return formatted_result



@app.route('/', methods=['GET', 'POST'])
def home():
    global top_n_indices, vec_q
    query = None
    formatted_results = []

    if request.method == 'POST':
        query = request.form.get('search')
        if query:
            vec_q = engine.query_to_vector(query)
            top_n_indices, similarities = engine.search(vec_q, n=10)

            formatted_results = format_results(similarities)

    return render_template('index.html', risultati=formatted_results, query=query)
        
@app.route('/submit-feedback', methods=['POST'])
def handle_feedback():
    global top_n_indices, vec_q
    
    id_string = request.form.get('feedback_ids', '')

    relevant_input = id_string.strip().split()
    relevant_doc_ids = [int(d) for d in relevant_input]

    vec_q = engine.apply_relevance_feedback(vec_q, relevant_doc_ids, top_n_indices)
    top_n_indices, similarities = engine.search(vec_q, n=10)

    formatted_results = format_results(similarities)

    return render_template('index.html', risultati=formatted_results)
    
    

if __name__ == '__main__':
    app.run(debug=True)