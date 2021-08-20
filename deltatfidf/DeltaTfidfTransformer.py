import scipy.sparse as sp
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class DeltaTfidfTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X_pos, X_neg, y=None):
        """Learn the idf vector (global term weights).
        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        y : None
            This parameter is not needed to compute tf-idf.
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
            frac_pos_neg = n_samples_pos / n_samples_neg
            idf = np.log(frac_pos_neg * df_neg / df_pos)

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
