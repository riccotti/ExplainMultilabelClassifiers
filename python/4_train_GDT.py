import gzip
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from multilabelexplanations import distance_functions
from multilabelexplanations import rules

warnings.filterwarnings("ignore")

columns_ylist = {'woman': 'service', 'yeast': 'Class', 'medical':'Class'}

dataset_list = ['yeast', 'woman','medical']
blackbox_list = ['rf', 'svm', 'mlp']

dt_params = {
    'max_depth': [None, 10, 20, 30, 40, 50, 70, 80, 90, 100],
    'min_samples_split': [2**i for i in range(1, 10)],
    'min_samples_leaf': [2**i for i in range(1, 10)],
}
cv = 5

dataset_list = ['medical']
blackbox_list = ['rf', 'svm', 'mlp']

for dataset in dataset_list:
    print('using dataset %s_2e.csv' % dataset)
    df_2e = pd.read_csv('../dataset/%s_2e.csv' % dataset)
    cols_Y = [col for col in df_2e.columns if col.startswith(columns_ylist[dataset])]
    cols_X = [col for col in df_2e.columns if col not in cols_Y]

    X2e = df_2e[cols_X].values
    y2e = df_2e[cols_Y].values

    for blackbox_name in blackbox_list:
        bb = pickle.load(gzip.open('../models/tuned_%s_%s.pickle.gz' % (blackbox_name, dataset), 'rb'))
        bb_labels = bb.predict(X2e)

        # traino un GDT facendo tuning degli iperparametri
        dt = DecisionTreeClassifier()
        sop = np.prod([len(v) for k, v in dt_params.items()])
        n_iter_search = min(100, sop)
        random_search = RandomizedSearchCV(dt, param_distributions=dt_params, scoring='f1_micro', n_iter=n_iter_search,
                                           cv=cv)
        random_search.fit(X2e, bb_labels)
        best_params = random_search.best_params_
        dt.set_params(**best_params)
        # fitto il GDT sul dataset
        dt.fit(X2e, bb_labels)
        GDT_labels = dt.predict(X2e)
        print('F1 score (fidelity) su X2E: %.3f' % f1_score(bb_labels, GDT_labels, average='micro'))
        # salvo il modello GDT fittato
        joblib.dump(dt, '../global_dt/GDT_to_mimic_%s_%s.pkl' % (blackbox_name, dataset))

        for id_i2e, i2e in enumerate(X2e):
            rule, len_rule = rules.istance_rule_extractor(i2e.reshape(1, -1), dt, cols_X)
            jrow = {
                'dataset_name': dataset,
                'bb_name': blackbox_name,
                'i2e_id': str(id_i2e),
                'i2e_bb_label': bb_labels[id_i2e].astype(int).tolist(),
                'i2e_GDT_label': GDT_labels[id_i2e].astype(int).tolist(),
                'hit_sm': distance_functions.simple_match_distance(bb_labels[id_i2e], GDT_labels[id_i2e]),
                'hit_jc': jaccard_similarity_score(bb_labels[id_i2e], GDT_labels[id_i2e]),
                'rule': rule,
                'len_rule': str(len_rule),
            }

            try:
                json_str = ('%s\n' % json.dumps(jrow)).encode('utf-8')
            except Exception as e:
                print('Problems in dumping row: ' + e)
                break
            try:
                with gzip.GzipFile('../global_dt/GDT_to_mimic_%s_%s_metrics.json.gz' % (dataset, blackbox_name),
                                   'a') as fout:
                    fout.write(json_str)
            except Exception as e:
                print('Problems in saving the output: ' + e)
                break