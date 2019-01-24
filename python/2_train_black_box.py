import datetime
import gzip
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

black_box_list = ['rf','svm','mlp']
dataset_list = ['yeast','woman','medical']
columns_ylist = {'woman': 'service', 'yeast': 'Class', 'medical':'Class'}

black_box_list = {
    'rf': RandomForestClassifier(),
    'svm': OneVsRestClassifier(LinearSVC()),
    'mlp': MLPClassifier(),
}

parameter = {
    'rf': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'n_jobs': 4,
    },
    'svm': {
        'penalty': 'l2',
        'dual': False,
        'C': 1.0,
    },
    'mlp': {
        'hidden_layer_sizes': (256, 128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'early_stopping': True,

    }
}

parameter_rs = {
    'rf': {
        'n_estimators': [100],
        'max_depth': [None, 10, 20, 30, 40, 50, 70, 80, 90, 100],
        'min_samples_split': [2 ** i for i in range(1, 10)],
        'min_samples_leaf': [2 ** i for i in range(1, 10)],
        'max_features': ['auto'],
        'n_jobs': [4],
    },
    'svm': {
        'estimator__penalty': ['l1', 'l2'],
        'estimator__dual': [False],
        'estimator__C': [0.001, 0.01, 0.1, 1.0, 2.0, 4.0, 8.0],
    },
    'mlp': {
        'hidden_layer_sizes': [(100,), (128, 64), (256, 128, 64), (512, 128)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01],
        'early_stopping': [True],

    }
}

import warnings

warnings.filterwarnings("ignore")

cv = 5
for idx, dataset in enumerate(dataset_list):
    print(datetime.datetime.now(), 'dataset: %s' % dataset)
    df_bb = pd.read_csv('../dataset/%s_bb.csv' % dataset)
    df_2e = pd.read_csv('../dataset/%s_2e.csv' % dataset)

    cols_Y = [col for col in df_bb.columns if col.startswith(columns_ylist[dataset])]
    cols_X = [col for col in df_bb.columns if col not in cols_Y]

    X = df_bb[cols_X].values
    y = df_bb[cols_Y].values

    X2e = df_2e[cols_X].values
    y2e = df_2e[cols_Y].values

    for blackbox_name in black_box_list:
        print(datetime.datetime.now(), '\tblack box: %s' % blackbox_name)

        params = parameter_rs[blackbox_name]
        bb = black_box_list[blackbox_name]
        sop = np.prod([len(v) for k, v in params.items()])
        n_iter_search = min(100, sop)
        random_search = RandomizedSearchCV(bb, param_distributions=params,
                                           scoring='f1_micro', n_iter=n_iter_search, cv=cv)
        random_search.fit(X, y)
        best_params = random_search.best_params_
        bb.set_params(**best_params)

        # params = parameter[blackbox_name]
        # bb = OneVsRestClassifier(bb) if blackbox_name == 'svm' else bb
        # bb = black_box_list[blackbox_name]()
        # bb.set_params(**params)
        # bb = OneVsRestClassifier(bb) if blackbox_name == 'svm' else bb

        bb.fit(X, y)
        pred_bb = bb.predict(X)
        pred_2e = bb.predict(X2e)
        print(datetime.datetime.now(), '\t  F1 - train: %.4f' % f1_score(y, pred_bb, average='micro'))
        print(datetime.datetime.now(), '\t  F1 -  test: %.4f' % f1_score(y2e, pred_2e, average='micro'))

        pickle_file = gzip.open('../models/tuned_%s_%s.pickle.gz' % (blackbox_name, dataset), 'wb')
        pickle.dump(bb, pickle_file)
        pickle_file.close()



###### IF YOU HAVE YOUR ALREADY TRAINED BLACK BOXES UNCOMMENT BELOW
# for idx, dataset in enumerate(dataset_list):
#     df_bb = pd.read_csv('../dataset/%s_bb.csv' % dataset)
#     df_2e = pd.read_csv('../dataset/%s_2e.csv' % dataset)
#
#     cols_Y = [col for col in df_bb.columns if col.startswith(columns_ylist[dataset])]
#     cols_X = [col for col in df_bb.columns if col not in cols_Y]
#
#     X = df_bb[cols_X].values
#     y = df_bb[cols_Y].values
#
#     X2e = df_2e[cols_X].values
#     y2e = df_2e[cols_Y].values
#
#     print(dataset)
#     for blackbox_name in black_box_list:
#         bb = pickle.load(gzip.open('../models/tuned_%s_%s.pickle.gz' % (blackbox_name, dataset), 'rb'))
#
#         pred_bb = bb.predict(X)
#         pred_2e = bb.predict(X2e)
#         print('%s:\t F1 - train: %.4f' % (blackbox_name, f1_score(y, pred_bb, average='micro')))
#         print('%s:\t F1 -  test: %.2f' % (blackbox_name, round(f1_score(y2e, pred_2e, average='micro'),2)))
#
#     print('')