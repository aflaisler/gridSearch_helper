from __future__ import print_function
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterSampler, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_union
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression, make_classification
from six import string_types
from pprint import pprint
from time import time
import logging
import numpy as np


X, y, coef = make_regression(n_samples=1000, n_features=2,
                             n_informative=1, noise=10,
                             coef=True, random_state=0)


def get_data(filename, n=None):
    df = pd.read_csv(filename)
    X, y = df.body[:n], df.section_name[:n]
    return X, y


pipeline = make_pipeline(StandardScaler(), Lasso())

# parameters = {'multinomialnb__alpha': [.3, .4, .5],
#               'multinomialnb__fit_prior': [True, False],
#               'tfidfvectorizer__lowercase': [True, False]}
parameters = dict()

# d2


# if __name__ == '__main__':
# X, y = get_data('./data/articles.csv', n=1000)
# y
# we define the gridsearchCV
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, cv=2)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)
t0 = time()
# we fit it
grid_search.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
# get the best parameters
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


d = pipeline.get_params()

del d['steps']

keys_ = [k for k in d.keys() if "__" in k]

keys_

# keys_
d2 = {}
for k in keys_:
    # print "penalty" in k
    if ("dual" in k) or ("intercept" in k):
        next
    elif (("lasso" in k.lower()) or ("ridge" in k.lower())) and ("alpha" in k):
        d2.update({k: np.arange(0.01, 1.0, 0.05)})
    elif "max_iter" in k:
        d2.update({k: [o for o in np.multiply([1, 5, 10], 100)]})
    elif isinstance(d[k], string_types):
        d2.update({k: [d[k].encode('utf8')]})
        if ("penalty" in k) and ('l' in d[k]):
            d2.update({k: ['l1', 'l2']})

    elif "n_jobs" in k:
        d2.update({k: [-1]})

    elif isinstance(d[k], tuple):
        d2.update({k: [(1, i) for i in range(1, 4)]})
    elif isinstance(d[k], bool):
        d2.update({k: [b for b in (True, False)]})
    elif ("max_df" in k) and isinstance(d[k], float) and d[k] == 1:
        d2.update({k: list(np.round(np.linspace(0.5, 1, 3), 2))})
    elif isinstance(d[k], int) and d[k] == 1:
        d2.update({k: [1, 2, 4, 6]})
    else:
        d2.update({k: [d[k]]})

d2

"Lasso"

np.arange(0.01, 1.0, 0.005)

# redo the grid_search with the generation of the hyperparams
parameters = d2

# we define the gridsearchCV
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, cv=2)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)
t0 = time()
# we fit it
grid_search.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
# get the best parameters
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
