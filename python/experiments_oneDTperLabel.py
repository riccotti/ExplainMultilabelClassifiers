import sys
import os
import pandas as pd
import numpy as np
import pickle
import datetime
import gzip
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance
from multilabelexplanations import distance_functions
from multilabelexplanations import rules
from multilabelexplanations import synthetic_neighborhood
import json

dataset = sys.argv[1]
blackbox_name = sys.argv[2]
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

#dict with key: dataset_name and value target columns names
columns_ylist = {'woman': 'service', 'yeast': 'Class', 'diabete':'diag_','medical':'Class'}

#dict with key: dataset_name and value list of lists, list[0] = countinuous var names, list[1] = discrete var names
with open('../dataset/dict_names.pickle', 'rb') as handle:
    columns_type_dataset = pickle.load(handle)

try:
    df_2e = pd.read_csv('../dataset/%s_2e.csv' % dataset)
    #print('Using dataset %s',dataset)
    #print('number of instances in the dataset: %s' % len(df_2e))
except Exception as e:
    print("Problem in loading the dataset: %s" % e)


cols_Y = [col for col in df_2e.columns if col.startswith(columns_ylist[dataset])]
cols_X = [col for col in df_2e.columns if col not in cols_Y]

X2e = df_2e[cols_X].values
cols_Y_BB = ['BB_'+str(col) for col in cols_Y]
 
#print('Using black box: %s' % blackbox_name)    
try:
    bb = pickle.load(gzip.open('/home/user/venvs/LORE_env/ExplainMultilabelClassifiers/models/tuned_%s_%s.pickle.gz' % (blackbox_name, dataset), 'rb'))
except Exception as e:
    print("Problem loading the model:\n %s" % e)
    
try:
    y_bb = bb.predict(X2e)
except Exception as e:
    print("Problem with trained model, probably features do not match those used to train the model:\n %s" % e)
    
#Creating the dataset to be explained X2E
BB_predictions_df = pd.DataFrame(y_bb,columns=cols_Y_BB)
Xtest_features_df = pd.DataFrame(X2e,columns=cols_X)
X2E = pd.concat([Xtest_features_df,BB_predictions_df],axis=1)

dt_params = {
    'max_depth': [None, 10, 20, 30, 40, 50, 70, 80, 90, 100],
    'min_samples_split': [2**i for i in range(1, 10)],
    'min_samples_leaf': [2**i for i in range(1, 10)],
}


#importing distances matrices
pdist_matrix_squared = pd.read_csv("../dataset/%s_featspace_pdist.csv" % dataset, header=None).values
pdist_matrix_label_squared = pd.read_csv("../dataset/%s_%s_labelspace_pdist.csv" % (dataset,blackbox_name),header=None).values
pdist_matrix_featlabel_squared = pd.read_csv("../dataset/%s_%s_featlabelspace_pdist.csv" % (dataset,blackbox_name),header=None).values

for instance in X2E.index.values:
    
    #instance to be explained
    i2e = X2E.loc[instance] 
    i2e_values = i2e[cols_X].values
    y_i2e_bb = bb.predict(i2e_values.reshape(1, -1))
    
    #number of real neighbors to consider:
    k = int(0.5*np.sqrt(len(X2E))) 
    #number of synthetic neighbors to generate
    size= 1000 

    #sample kNN for synthetic neighborhood based on feature space distances
    sampleKnn_feat_space = X2E.loc[pd.DataFrame(pdist_matrix_squared).loc[instance].sort_values().index.values[0:k]]
    #sample kNN for synthetic neighborhood based on label space distances
    sampleKnn_label_space = X2E.loc[pd.DataFrame(pdist_matrix_label_squared).loc[instance].sort_values().index.values[0:k]]
    #sample kNN for synthetic neighborhood based on mixed space distances 
    sampleKnn_mixed_space = X2E.loc[pd.DataFrame(pdist_matrix_featlabel_squared).loc[instance].sort_values().index.values[0:k]]

    #####################################################################################################
    ###############################MIXED NEIGHBORHOOD####################################################

    alpha_beta_sample_knn = synthetic_neighborhood.sample_alphaFeat_betaLabel(sampleKnn_feat_space,sampleKnn_label_space,alpha=0.7,beta=0.3)
    synthetic_neighborhood1 = synthetic_neighborhood.random_synthetic_neighborhood_df(
        alpha_beta_sample_knn,size,
        discrete_var=X2E[columns_type_dataset[dataset][1]].columns.values,
        continuous_var=X2E[columns_type_dataset[dataset][0]].columns.values,
        classes_name=cols_Y_BB)
    BB_label_syn1_df = pd.DataFrame(bb.predict(synthetic_neighborhood1.values),columns=cols_Y_BB)
    BB_label_kNN1_df = pd.DataFrame(bb.predict(alpha_beta_sample_knn[cols_X].values),columns=cols_Y_BB)
    synthetic_neighborhood1 = pd.concat([synthetic_neighborhood1,BB_label_syn1_df],1)

    #####################################################################################################
    ###############################UNIFIED NEIGHBORHOOD####################################################
    synthetic_neighborhood2 = synthetic_neighborhood.random_synthetic_neighborhood_df(sampleKnn_mixed_space,\
                                                                                      size,\
                                                                                      discrete_var=X2E[columns_type_dataset[dataset][1]].columns.values,\
                                                                                      continuous_var=X2E[columns_type_dataset[dataset][0]].columns.values,\
                                                                                      classes_name=cols_Y_BB)
    #uso la black box per assegnare una predizione ad ogni istanza sintetica
    BB_label_syn2_df = pd.DataFrame(bb.predict(synthetic_neighborhood2.values),columns=cols_Y_BB)
    BB_label_kNN2_df = pd.DataFrame(bb.predict(sampleKnn_mixed_space[cols_X].values),columns=cols_Y_BB)
    synthetic_neighborhood2 = pd.concat([synthetic_neighborhood2,BB_label_syn2_df],1)

    #building one tuned DT for each label on synthetic neighborhood1 (unified) and extracting rules
    unionDT_labels_syn1, unionDT_labels_knn1, unionDT_labels_i2e1, DT_rules1, DT_rules_len1 = synthetic_neighborhood.oneDTforlabelpreds(synthetic_neigh = synthetic_neighborhood1,\
                                                                                                                                        knn_neigh = alpha_beta_sample_knn,\
                                                                                                                                        i2e = i2e,\
                                                                                                                                        cols_X = cols_X,\
                                                                                                                                        cols_Y_BB = cols_Y_BB,\
                                                                                                                                        param_distributions=dt_params)
    #building one tuned DT for each label on synthetic neighborhood2 (mixed) and extracting rules
    unionDT_labels_syn2, unionDT_labels_knn2, unionDT_labels_i2e2, DT_rules2, DT_rules_len2 = synthetic_neighborhood.oneDTforlabelpreds(synthetic_neigh = synthetic_neighborhood2,\
                                                                                                                                        knn_neigh = sampleKnn_mixed_space,\
                                                                                                                                        i2e = i2e,\
                                                                                                                                        cols_X = cols_X,\
                                                                                                                                        cols_Y_BB = cols_Y_BB,\
                                                                                                                                        param_distributions=dt_params)
    #evaluating the trees
    f1_syn1_tree1 = f1_score(y_true=BB_label_syn1_df.values,y_pred=unionDT_labels_syn1.values, average='micro')
    f1_kNN1_tree1 = f1_score(y_true=BB_label_kNN1_df.values,y_pred=unionDT_labels_knn1.values, average='micro')

    f1_syn2_tree2 = f1_score(y_true=BB_label_syn2_df.values,y_pred=unionDT_labels_syn2.values, average='micro')
    f1_kNN2_tree2 = f1_score(y_true=BB_label_kNN2_df.values,y_pred=unionDT_labels_knn2.values, average='micro')

    #print('y_i2e_bb[0].shape: %s' %y_i2e_bb[0].shape)
    #print('unionDT_labels_i2e1.values[0].shape: %s' %unionDT_labels_i2e1.values[0].shape)
    hit_jc_tree1 = jaccard_similarity_score(y_i2e_bb[0],unionDT_labels_i2e1.values[0])
    hit_sm_tree1 = distance_functions.simple_match_distance(y_i2e_bb[0],unionDT_labels_i2e1.values[0])

    hit_jc_tree2 = jaccard_similarity_score(y_i2e_bb[0],unionDT_labels_i2e2.values[0])
    hit_sm_tree2 = distance_functions.simple_match_distance(y_i2e_bb[0],unionDT_labels_i2e2.values[0])

    jrow = {
        'dataset_name':dataset,
        'bb_name': blackbox_name,
        'i2e_id': str(instance),
        'fidelity_tree1_syn':f1_syn1_tree1,
        'fidelity_tree1_kNN':f1_kNN1_tree1,
        'fidelity_tree2_syn':f1_syn2_tree2,
        'fidelity_tree2_kNN':f1_kNN2_tree2,
        'i2e_bb_label': y_i2e_bb.astype(int).tolist(),
        'i2e_tree1_label':unionDT_labels_i2e1.values[0].astype(int).tolist(),
        'i2e_tree2_label':unionDT_labels_i2e2.values[0].astype(int).tolist(),
        'hit_sm_tree1':hit_sm_tree1,
        'hit_jc_tree1':hit_jc_tree1,
        'hit_sm_tree2':hit_sm_tree2,
        'hit_jc_tree2':hit_jc_tree2,
        'rules_trees1':DT_rules1,
        'rules_trees2':DT_rules2,
        'lenght_rules_trees1':DT_rules_len1,
        'lenght_rules_trees2':DT_rules_len2,
    }

    try:
        json_str = ('%s\n' % json.dumps(jrow)).encode('utf-8')
    except Exception as e:
        print('Problems in dumping row: %s' % e)

    try:
        with gzip.GzipFile('../output/%s_%s_%s_explanationsandmetrics_oneDTperLabel.json.gz' % (now, dataset, blackbox_name), 'a') as fout:
            fout.write(json_str)
    except Exception as e:
        print('Problems in saving the output: %s' % e)









