import numpy as np
from numpy import ndarray
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, X):
        self.X = X
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.loc[:, '']
        return X_new
    


class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins):
        self.bins = bins

    def fit(self, X, y=None):
        self.X_fit_ = X
        return self

    def transform(self, X, y=None):
        X_binned = pd.cut(X, bins=self.bins, labels=False)
        return X_binned.values.reshape(-1, 1)
…