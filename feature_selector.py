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
