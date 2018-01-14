

class Testy(object):

    def DecisionTreeClassifier_hp(self):
        return {'decisiontreeclassifier__class_weight': [None],
                'decisiontreeclassifier__criterion': ['gini', 'entropy'],
                'decisiontreeclassifier__max_depth': [None, 2, 3, 4, 6],
                'decisiontreeclassifier__max_features': [None, 2, 4, 6],
                'decisiontreeclassifier__max_leaf_nodes': [None, 2, 4, 6],
                'decisiontreeclassifier__min_impurity_decrease': [0.0, .1, .2],
                'decisiontreeclassifier__min_impurity_split': [None],
                'decisiontreeclassifier__min_samples_leaf': [1, 2, 4, 6],
                'decisiontreeclassifier__min_samples_split': [2, 4, 6],
                'decisiontreeclassifier__min_weight_fraction_leaf': [0.0],
                'decisiontreeclassifier__presort': [True, False],
                'decisiontreeclassifier__random_state': [None],
                'decisiontreeclassifier__splitter': ['best', 'random']}
#
#
# def RandomForestClassifier_hp():
#     return {'randomforestclassifier__bootstrap': [True, False],
#             'randomforestclassifier__class_weight': [None],
#             'randomforestclassifier__criterion': ['gini', 'entropy'],
#             'randomforestclassifier__max_depth': [None, 2, 4, 6],
#             'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
#             'randomforestclassifier__max_leaf_nodes': [None],
#             'randomforestclassifier__min_impurity_decrease': [0.0],
#             'randomforestclassifier__min_impurity_split': [None],
#             'randomforestclassifier__min_samples_leaf': [1, 2, 4, 6],
#             'randomforestclassifier__min_samples_split': [2],
#             'randomforestclassifier__min_weight_fraction_leaf': [0.0],
#             'randomforestclassifier__n_estimators': [10, 100, 1000],
#             'randomforestclassifier__oob_score': [True, False],
#             'randomforestclassifier__random_state': [None]}
#
#
# def standardscaler_hp():
#     return {'standardscaler__copy': [True, False],
#             'standardscaler__with_mean': [True, False],
#             'standardscaler__with_std': [True, False]}
#
#
# def svc_hp():
#     return {'svc__C': np.logspace(-5, 0, 100),
#             'svc__cache_size': [200],
#             'svc__class_weight': [None],
#             'svc__coef0': [0.0],
#             'svc__decision_function_shape': ['ovr'],
#             'svc__degree': [3],
#             'svc__gamma': ['auto'],
#             'svc__kernel': ['rbf', 'linear', 'poly', 'rbf', 'sigmoid'],
#             'svc__max_iter': [100, 500, 1000],
#             'svc__probability': [True, False],
#             'svc__random_state': [None],
#             'svc__shrinking': [True, False],
#             'svc__tol': [0.001]}
#
#
# RandomForestClassifier_hp()
# #
# # pca
# #
# # one_hot_encoder
# #
# # standard_scaler
# # min_max_scaler
# # normalizer
# #
# # ts_lagselector
# #
# # tfidf
