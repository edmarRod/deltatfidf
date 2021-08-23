# Deltatfidf

Module for using [Delta Tf-idf](https://ebiquity.umbc.edu/_file_directory_/papers/446.pdf) based on scikit-learn's Tf-idf.

``` python
>>> from deltatfidf import DeltaTfidfVectorizer
>>> corpus = [
... "I like cheese",
... "Cheese is the best",
... "I do not like cheese",
... "Cheese stinks"
... ]
>>> labels = [0,0,1,1]
>>> vectorizer = DeltaTfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus, labels)
>>> vectorizer.get_feature_names()
['best', 'cheese', 'do', 'is', 'like', 'not', 'stinks', 'the']
```

Classes must be labeled as either 0 or 1.

## Installation

To install use:

``` bash
pip install git+https://github.com/edmarRod/deltatfidf
```
