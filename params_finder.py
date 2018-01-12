import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterSampler, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_union
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from six import string_types

pipeline = make_pipeline(CountVectorizer(), LogisticRegression())

parameters = {'multinomialnb__alpha': [.3, .4, .5],
              'multinomialnb__fit_prior': [True, False],
              'tfidfvectorizer__lowercase': [True, False]}
# parameters = d2
parameters = {'countvectorizer__analyzer': ['word'],
              'countvectorizer__binary': [True, False],
              'countvectorizer__decode_error': ['strict'],
              # 'countvectorizer__dtype': [np.int64],
              # 'countvectorizer__encoding': ['utf-8'],
              # 'countvectorizer__input': ['content'],
              # 'countvectorizer__lowercase': [True, False],
              # 'countvectorizer__max_df': [.5, 1.0],
              # 'countvectorizer__max_features': [None],
              # 'countvectorizer__min_df': [1, 2, 4, 6],
              # 'countvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
              # 'countvectorizer__preprocessor': [None],
              # 'countvectorizer__stop_words': [None],
              # 'countvectorizer__strip_accents': [None],
              # 'countvectorizer__token_pattern': ['(?u)\\b\\w\\w+\\b'],
              # 'countvectorizer__tokenizer': [None],
              # 'countvectorizer__vocabulary': [None],
              # 'logisticregression__C': [1.0],
              # 'logisticregression__class_weight': [None],
              # 'logisticregression__dual': [True, False],
              'logisticregression__fit_intercept': [True, False],
              'logisticregression__intercept_scaling': [1],
              'logisticregression__max_iter': [100],
              'logisticregression__multi_class': ['ovr'],
              'logisticregression__n_jobs': [-1],
              'logisticregression__penalty': ['l1', 'l2'],
              'logisticregression__random_state': [None],
              'logisticregression__tol': [0.0001],
              'logisticregression__verbose': [0],
              'logisticregression__warm_start': [True, False]}

# d2


def get_data(filename, n=None):
    df = pd.read_csv(filename)
    X, y = df.body[:n], df.section_name[:n]
    return X, y


# if __name__ == '__main__':
X, y = get_data('./data/articles.csv', n=1000)
# y
# we define the gridsearchCV
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, cv=2)

# print("Performing grid search...")
# print("pipeline:", [name for name, _ in pipeline.steps])
# print("parameters:")
# print(parameters)
# t0 = time()
# we fit it
grid_search.fit(X, y)
# print("done in %0.3fs" % (time() - t0))
# print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
# get the best parameters
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


d = pipeline.get_params()
d
del d['steps']
d
keys_ = [k for k in d.keys() if "__" in k]

keys_
d2 = {}
for k in keys_:
    # print "penalty" in k
    if ("dual" in k) or ("intercept" in k):
        next
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

n = 100
np.multiply([1, 5, 10], 100)

[o for o in np.multiply([1, 5, 10], 100)]


def helperTuple(tup):
    return [(1, i) for i in range(1, 4)]


list(np.round(np.linspace(0.5, 1, 3), 2))

ls = ['l2']

"l" in ls[0]
isinstance(1.0, float)
dhelperTuple((1, 1))

list(np.round(np.linspace(0, 1, 10), 2))
