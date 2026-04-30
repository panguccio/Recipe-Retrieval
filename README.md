# Recipe Retrieval - An Information Retrieval System

> Explanatory pipeline of the system in `recipe_retrieval.ipynb`

This repository defines an Information Retrieval System that can be used for recipe datasets. Given a phrase query, it will return the recipes that are more relevant in the current dataset, which contains approximately 60 thousands documents.

The recipes, in `.json` format, are pre-processed and then saved as matrixes of **TF-TDF** values, coherently with the Vector Space Model. The words are normalized and saved as an **inverted index**: each term is mapped to a posting list, which contains the document-ids of the recipes containing that term.

The inverted index is built in such a way that each term is formed by the couple `word` and `zone`. This choice was made in order to give much more importance in the retrieval to words contained in titles with respect to the ones in the ingredients and instructions parts.

For the actual retrieval, the `search engine` transforms the queries into vectors and calculates the cosine similarity with the documents vectors. The ones with larger cosine similarity are returned, ordered. The user can also provide a **relevance feedback**, that will transpose the query in the vector space closer to the relevant documents, using the Rocchio algorithm.

To make these calculations efficient, given the large number of documents and terms, the `scipy` sparce arrays were used. This made the cosine similarity calculation trivial and lowered significantly the amount of space needed to save the matrix, because of the large number of 0-values in the vectors.

A test benchmark was also developed to test our system on a subset of recipes, based on automatically generated queries. The system gave strong results and proved its robustness even with original phrase queries.

Developed by: Gabriele Pasqualini, Federico Marenco and Anna Guccione.


## Project Structure

```text
recipe-retrieval/
├── data/                       
│   ├── corpus/                 # * corpus of recipes
│   └── bin/                    # * .pkl of the inverted index and document vectors
│   └── benchmark/              # * data used for the benchmark evaluation
├── src/                        
│   ├── core/                  
│   │   ├── models.py           # Term, Posting, PostingList classes
│   │   └── index.py            # InvertedIndex class
│   ├── search/                 
│   │   ├── engine.py           # SearchEngine class
│   │   └── cli.py              # retrieval logic for command line interface
│   │   └── app.py              # retrieval logic with a webapp
│   ├── utils/                  
│   │   ├── tokenizer.py        # tokenizing logic
│   │   ├── build_index.py      # index construction logic
│   │   └── corpus_parser.py    # corpus preprocessing logic
│   ├── benchmark/               
│   │   └── benchmark.py        # evaluation of the system
│   └── web/                    # integration with Flask
├── config.yaml                 # parameters
├── requirements.txt
└── README.md
```