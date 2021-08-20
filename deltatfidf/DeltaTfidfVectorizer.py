import numpy as np
from collections import Counter
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from sklearn.feature_extraction.text import TfidfVectorizer

from DeltaTfidfTransformer import DeltaTfidfTransformer


class DeltaTfidfVectorizer(CountVectorizer):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._tfidf = DeltaTfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        """Norm of each row output, can be either "l1" or "l2"."""
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        """Whether or not IDF re-weighting is used."""
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        """Whether or not IDF weights are smoothed."""
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        """Whether or not sublinear TF scaling is applied."""
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        """Inverse document frequency vector, only defined if `use_idf=True`.
        Returns
        -------
        ndarray of shape (n_features,)
        """
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal to vocabulary size = %d"
                    % (len(value), len(self.vocabulary))
                )
        self._tfidf.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def _check_binary(self, y):
        count_class = Counter(y)
        num_classes = len(count_class)
        if num_classes != 2:
            raise ValueError(f"Expected 2 classes for target, got {num_classes}")

        cnt_0 = count_class[0]
        cnt_1 = count_class[1]

        if (cnt_0 == 0) | (cnt_1 == 0):
            raise ValueError(
                f"Expected classes to have more than 0 elements, got class 0:{cnt_0}, class 1:{cnt_1}. Check if classes are set as 0 and 1."
            )

    def fit(self, raw_documents, y):
        """Learn vocabulary and idf from training set.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        y : Array with classes as 0 or 1.
            Array with the binary class.
        Returns
        -------
        self : object
            Fitted vectorizer.
        """

        self._check_binary(y)
        self._check_params()
        self._warn_for_unused_params()
        cnt_vect = super().fit_transform(raw_documents)
        X_pos = cnt_vect[y == 1]
        X_neg = cnt_vect[y == 0]
        self._tfidf.fit(X_pos, X_neg, y)
        return self

    def fit_transform(self, raw_documents, y):
        """Learn vocabulary and idf, return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        y : Array with classes as 0 or 1.
            Array with the binary class.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, msg="The Delta TF-IDF vectorizer is not fitted")

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

    def _more_tags(self):
        return {"X_types": ["string"], "_skip_test": True}
