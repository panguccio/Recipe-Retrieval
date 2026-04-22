# IR-System
Project for Information Retrieval exam.

How to run:
1. clone the repository
```bash
git clone https://github.com/panguccio/RecipeRetrieval.git
cd RecipeRetrieval
```
2. Install the dependencies
```bash
python3 -m pip install -r requirements.txt
```
3. Launch `testing.py`
# Osservaizoni
    1. Non considerare ingredienti, nella tokenizzazione
    2. Fare due tokenizzazioni una per titolo una per instruction
    3. Deciso di sostituire il trattino con uno spazio vuoto
    4. Togliere tutto che non sia lettere minuscole e maiuscole (Trattini, Numeri, ....)
    5. Togliere le stop-word con il TF IDF 
    6. Abbiamo aggiunto una mapatura del pos utilizzando il tagger e il lemmatizzatore di nltk 
    7. Abbiamo deciso di non ricavarci subito l'IDF ma di calmcolarlo dopo una volta che la posting-list è completa 