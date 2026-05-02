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
query_vec = None

def format_results(similarity):
    global top_n_indices, query_vec
    formatted_results = []
    
    for doc_id in top_n_indices:
        recipe = corpus.get(str(doc_id), {})
        
        raw_ingredients = recipe.get('ingredients', [])

        ingredients_list = [line.strip() for line in raw_ingredients.splitlines() if line.strip()]

        formatted_results.append({
            'id': doc_id,
            'title': recipe.get('title', 'Title not found'),
            'score': round(float(similarity[doc_id]), 4),
            'ingredients': ingredients_list,
            'instructions': recipe.get('instructions', [])
        })

    return formatted_results



@app.route('/', methods=['GET', 'POST'])
def home():
    global top_n_indices, query_vec
    query = None
    formatted_results = []

    if request.method == 'POST':
        query = request.form.get('search')
        if query:
            query_vec = engine.query_to_vector(query)
            top_n_indices, similarities = engine.search(query_vec, n=10)

            formatted_results = format_results(similarities)

    return render_template('index.html', results=formatted_results, query=query)
        
@app.route('/submit-feedback', methods=['POST'])
def handle_feedback():
    global top_n_indices, query_vec
    
    id_string = request.form.get('feedback_ids', '')

    relevant_input = id_string.strip().split()
    relevant_doc_ids = [int(d) for d in relevant_input]

    query_vec = engine.apply_relevance_feedback(query_vec, relevant_doc_ids, top_n_indices)
    top_n_indices, similarities = engine.search(query_vec, n=10)

    formatted_results = format_results(similarities)

    return render_template('index.html', results=formatted_results)
    
    

if __name__ == '__main__':
    app.run(debug=True)