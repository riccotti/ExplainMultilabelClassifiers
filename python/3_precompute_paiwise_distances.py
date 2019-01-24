import pandas as pd
import numpy as np
import pickle
import gzip
from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance
from multilabelexplanations import distance_functions


def mixed_distance(x, y, n_var_cont, cdist, ddist):
    # type: (pandas.Series, pandas.Series, list, list, list, function, function) -> double
    """
    This function return the mixed distance between instance x and instance y
    :param x: np.array, instance 1
    :param y: np.array, instance 2
    :param discrete: slices dicrete
    :param continuous: slices continuos
    :param ddist: function, distance function for discrete variables
    :param cdist: function, distance function for continuos variables
    :return: double
    """
    wc = 0.
    wd = 0.
    cd = 0.
    dd = 0.
    n_var_disc = len(x[n_var_cont:])

    if n_var_cont != 0:
        wc = n_var_cont / (n_var_cont + n_var_disc)
        xc = x[0:n_var_cont]
        yc = y[0:n_var_cont]
        cd = cdist(xc, yc)

    if n_var_disc != 0:
        wd = n_var_disc / (n_var_cont + n_var_disc)
        xd = x[n_var_cont:]
        yd = y[n_var_cont:]
        dd = ddist(xd, yd)

    return wd * dd + wc * cd


black_box_list = ['rf','svm','mlp']
dataset_list = ['yeast','woman','medical']

#lista contenente i prefissi delle colonne dei dataframe che contengono le classi da predire
columns_ylist = {'woman': 'service', 'yeast': 'Class','medical':'Class'}

#dizionario con chiave nome del dataset e valore una lista di liste, lista[0] = nomi var continue, lista[1] = nomi var discrete
with open('../dataset/dict_names.pickle', 'rb') as handle:
    columns_type_dataset = pickle.load(handle)


def create_dist_func(dataset):
    mydist = lambda x, y: mixed_distance(x, y, n_var_cont=len(columns_type_dataset[dataset][0]),
                                         cdist=distance_functions.normalized_euclidean_distance,
                                         ddist=distance.hamming)
    return mydist


for dataset in dataset_list:

    mydist = create_dist_func(dataset)

    try:
        df_2e = pd.read_csv('../dataset/%s_2e.csv' % dataset)
        print('Using dataset %s' % dataset)
        print('number of instances in the dataset: %s' % len(df_2e))
    except Exception:
        print("Problem in loading the dataset")

    cols_Y = [col for col in df_2e.columns if col.startswith(columns_ylist[dataset])]
    cols_X = [col for col in df_2e.columns if col not in cols_Y]

    X2e = df_2e[cols_X].values
    cols_Y_BB = ['BB_' + str(col) for col in cols_Y]

    for blackbox_name in black_box_list:

        try:
            bb = pickle.load(gzip.open('../models/tuned_%s_%s.pickle.gz' % (blackbox_name, dataset), 'rb'))
        except Exception as e:
            print("Problem loading the model: " + e)
            break

        try:
            y_bb = bb.predict(X2e)
        except Exception as e:
            print("Problem with trained model, probably features do not match those used to train the model: " + e)
            break

        # Creating the dataset to be explained X2E
        BB_predictions_df = pd.DataFrame(y_bb, columns=cols_Y_BB)
        Xtest_features_df = pd.DataFrame(X2e, columns=cols_X)
        X2E = pd.concat([Xtest_features_df, BB_predictions_df], axis=1)
        X2E_len = len(X2E)
        X2E_no_class = X2E[columns_type_dataset[dataset][0] + columns_type_dataset[dataset][1]]
        X2E_sorted = X2E[columns_type_dataset[dataset][0] + columns_type_dataset[dataset][1] + cols_Y_BB]

        print('pairwise distance in the label space, %s' % blackbox_name)
        pdist_matrix_label = pdist(X2E[cols_Y_BB].values, distance.hamming)
        pdist_matrix_label_squared = squareform(pdist_matrix_label)
        np.savetxt("../dataset/%s_%s_labelspace_pdist.csv" % (dataset, blackbox_name), pdist_matrix_label_squared,
                   delimiter=",")

        if dataset != 'medical':
            print('pairwise distance in the feature+label space, %s' % blackbox_name)
            pdist_matrix_featlabel = pdist(X2E_sorted.values, mydist)
            pdist_matrix_featlabel_squared = squareform(pdist_matrix_featlabel)
            np.savetxt("../dataset/%s_%s_featlabelspace_pdist.csv" % (dataset, blackbox_name),
                       pdist_matrix_featlabel_squared, delimiter=",")
        else:
            print('pairwise distance in the feature+label space (only discrete), %s' % blackbox_name)
            pdist_matrix_featlabel = pdist(X2E_sorted.values, distance.hamming)
            pdist_matrix_featlabel_squared = squareform(pdist_matrix_featlabel)
            np.savetxt("../dataset/%s_%s_featlabelspace_pdist.csv" % (dataset, blackbox_name),
                       pdist_matrix_featlabel_squared, delimiter=",")

    print('pairwise distance in the feature space')
    if (dataset == 'woman') or (dataset == 'diabete'):
        print('pairwise distance in the feature space (with discrete variables)')
        pdist_matrix = pdist(X2E_no_class.values, mydist)
        pdist_matrix_squared = squareform(pdist_matrix)
        np.savetxt("../dataset/%s_featspace_pdist.csv" % dataset, pdist_matrix_squared, delimiter=",")
    elif dataset == 'yeast':
        print('pairwise distance in the feature space (only continuous variables)')
        pdist_matrix = pdist(X2E_no_class.values, distance_functions.normalized_euclidean_distance)
        pdist_matrix_squared = squareform(pdist_matrix)
        np.savetxt("../dataset/%s_featspace_pdist.csv" % dataset, pdist_matrix_squared, delimiter=",")
    else:
        print('pairwise distance in the feature space (only discrete variables)')
        pdist_matrix = pdist(X2E_no_class.values, distance.hamming)
        pdist_matrix_squared = squareform(pdist_matrix)
        np.savetxt("../dataset/%s_featspace_pdist.csv" % dataset, pdist_matrix_squared, delimiter=",")
