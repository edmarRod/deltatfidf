import numpy as np

from ..DeltaTfidfVectorizer import DeltaTfidfVectorizer

data = ["word word2", "word word1 word2", "word1", "word1"]
label = [0, 0, 1, 1]

result_matrix = np.matrix(
    [
        [0.70710678, 0.0, 0.70710678],
        [0.6841916, -0.25251476, 0.6841916],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def test_vectorizer():
    dtfv = DeltaTfidfVectorizer()
    dtf_res = dtfv.fit_transform(data, label).todense()

    assert (np.round(dtf_res, 2) == np.round(result_matrix, 2)).all()
    assert len(dtfv.idf_) == len(dtfv.vocabulary_)
