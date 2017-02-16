#Textvectorizer

Python implementation of text vectorization algorithms.
Currently supported vectorization methods:
 - TFIDF-Vectorizer
 - LSI-Vectorizer

 For further explanation about the vectorization methods please refer the doc at
 https://github.com/jonysugianto/textvectorizer/doc/textvectorizer.pdf

requirements:
-python 3.5
-gensim
-nltk
-numpy
-scipy
-stop-words


 To install:
 1. git clone https://github.com/jonysugianto/textvectorizer

 2. cd textvectorizer/src

 3. set PYTHONPATH=./textvectorizer/src

 4. python exampe/example_build_dictionary.py
    create dictionary (needed for tfidf-vectorizer and lsi-vectorizer)

 5. python exampe/example_tfidf_vectorizer.py

 6. python exampe/example_lsi_vectorizer.py
