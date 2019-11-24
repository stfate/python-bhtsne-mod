# -*- coding: utf-8 -*-

"""
@package bhtsne_wrapper.py
@brief a wrapper class for python-bhtsne
@author Dan SASAI (YCJ,RDD)
"""

import scipy as sp
import sklearn.base
import bhtsne


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, n_components=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.n_components,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed,
            max_iter=self.max_iter
        )
