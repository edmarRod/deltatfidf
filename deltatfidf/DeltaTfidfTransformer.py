import scipy.sparse as sp
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class DeltaTfidfTransformer(TransformerMixin, BaseEstimator):
    """Transform a count matrix to a normalized tf or tf-idf representation.

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. The delta is because the difference between the tf*idf is
    taken between the 2 classes.
    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus. The difference is used to further differentiate and identify specific
    terms which identify the 2 classes.
    The formula that is used to compute the tf-idf for a term t of a document d
    in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
    computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
    n is the total number of documents in the document set and df(t) is the
    document frequency of t; the document frequency is the number of documents
    in the document set that contain the term t. The effect of adding "1" to
    the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely
    ignored. The formula for the delta is then simply the difference between
    the tf-idf of each class, trained on the class-specific corpsu.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as
    idf(t) = log [ n / (df(t) + 1) ]).
    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.
    Furthermore, the formulas used to compute tf and idf depend
    on parameter settings that correspond to the SMART notation used in IR
    as follows:
    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
    when ``norm=None``.
    Parameters
    ----------
    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
        - 'l2': Sum of squares of vector elements is 1. The cosine
          similarity between two vectors is their dot product when l2 norm has
          been applied.
        - 'l1': Sum of absolute values of vector elements is 1.
          See :func:`preprocessing.normalize`.
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting.
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    Attributes
    ----------
    idf_ : array of shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from deltatfidf import DeltaTfidfTransformer
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> import numpy as np
    >>> corpus = ['I like cheese',"Cheese is the best", "I do not like cheese","Cheese stinks"]
    >>> y = np.array([0,0,1,1])
    >>> cnt_vect = CountVectorizer().fit_transform(corpus)
    >>> X_pos = cnt_vect[y == 1]
    >>> X_neg = cnt_vect[y == 0]
    >>> dtf = DeltaTfidfTransformer().fit(X_pos, X_neg, y)
    >>> print(dtf.idf_)
    [ 0.69314718  0.         -0.69314718  0.69314718  0.         -0.69314718
     -0.69314718  0.69314718]
    """

    def __init__(self, *, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X_pos, X_neg, y=None):
        """Learn the idf vector (global term weights).
        Parameters
        ----------
        X_pos : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts for the positive class.
        X_neg : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts for the negative class.
        y : None
            This parameter is not needed to compute since the counts are passed.
        Returns
        -------
        self : object
            Fitted transformer.
        """

        X_pos = self._validate_data(X_pos, accept_sparse=("csr", "csc"))
        X_neg = self._validate_data(X_neg, accept_sparse=("csr", "csc"))

        if not sp.issparse(X_pos):
            X_pos = sp.csr_matrix(X_pos)
        if not sp.issparse(X_neg):
            X_neg = sp.csr_matrix(X_neg)

        dtype = X_pos.dtype if X_pos.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples_pos, n_features_pos = X_pos.shape
            n_samples_neg, n_features_neg = X_neg.shape

            # both should have the same vocabulary
            assert n_features_pos == n_features_neg

            df_pos = _document_frequency(X_pos)
            df_neg = _document_frequency(X_neg)

            df_pos = df_pos.astype(dtype, **_astype_copy_false(df_pos))
            df_neg = df_neg.astype(dtype, **_astype_copy_false(df_neg))

            # perform idf smoothing if required
            df_pos += int(self.smooth_idf)
            df_neg += int(self.smooth_idf)
            n_samples_pos += int(self.smooth_idf)
            n_samples_neg += int(self.smooth_idf)

            # no +1 since we want values with same idf to cancel
            # using log(x/y) == - log(y/x) so there are no zero divisions
            self.pos_idf = -1 * np.log(df_pos / n_samples_pos)
            self.neg_idf = -1 * np.log(df_neg / n_samples_neg)

            idf = self.pos_idf - self.neg_idf

            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features_pos, n_features_pos),
                format="csr",
            )

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation.
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        X = self._validate_data(
            X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"], msg="idf vector is not fitted")

            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def fit_transform(self, X, X_pos, X_neg, y=None):
        """Learn vocabulary and idf, return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.
        X_pos : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts for the positive class.
        X_neg : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts for the negative class.
        y : None
            This parameter is not needed to compute since the counts are passed.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self.fit(X_pos, X_neg, y)
        return self.transform(X, copy=False)

    @property
    def idf_(self):
        """Inverse document frequency vector, only defined if `use_idf=True`.
        Returns
        -------
        ndarray of shape (n_features,)
        """
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(
            value, diags=0, m=n_features, n=n_features, format="csr"
        )

    def _more_tags(self):
        return {"X_types": "sparse"}
