import sys
import os
import pandas as pd
import numpy as np
import pickle
import datetime
import gzip
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from scipy.spatial.distance import pdist,squareform
from sklearn.model_selection import RandomizedSearchCV
from scipy.spatial import distance


from multilabelexplanations import distance_functions
from multilabelexplanations import rules
from multilabelexplanations import synthetic_neighborhood
import json
import logging


dataset = sys.argv[1]
blackbox_name = sys.argv[2]
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

if not os.path.exists('../log'):
    os.makedirs('../log')

logFormat = '%(asctime)s - pid %(process)d - %(levelname)s -: %(message)s'
logFile = '../log/'+str(dataset)+'_'+str(blackbox_name)+'_experiments.log'
logger = logging.getLogger('checking_experiments_progress')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logFile, "a")
formatter = logging.Formatter(logFormat)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info('Starting the script')


#dict with key: dataset_name and value target columns names
columns_ylist = {'woman': 'service', 'yeast': 'Class', 'diabete':'diag_','medical':'Class'}

#dict with key: dataset_name and value list of lists, list[0] = countinuous var names, list[1] = discrete var names
with open('../dataset/dict_names.pickle', 'rb') as handle:
    columns_type_dataset = pickle.load(handle)

try:
    df_2e = pd.read_csv('../dataset/%s_2e.csv' % dataset)
    logger.info('Using dataset %s',dataset)
    logger.info('number of instances in the dataset: %s' % len(df_2e))
except Exception:
    logger.exception("Problem in loading the dataset")

cols_Y = [col for col in df_2e.columns if col.startswith(columns_ylist[dataset])]
cols_X = [col for col in df_2e.columns if col not in cols_Y]

X2e = df_2e[cols_X].values
cols_Y_BB = ['BB_'+str(col) for col in cols_Y]
 
logger.info('Using black box: %s' % blackbox_name)    
try:
    bb = pickle.load(gzip.open('../models/tuned_%s_%s.pickle.gz' % (blackbox_name, dataset), 'rb'))
except Exception:
    logger.exception("Problem loading the model")
    
try:
    y_bb = bb.predict(X2e)
except Exception:
    logger.exception("Problem with trained model, probably features do not match those used to train the model")
    
#Creating the dataset to be explained X2E
BB_predictions_df = pd.DataFrame(y_bb,columns=cols_Y_BB)
Xtest_features_df = pd.DataFrame(X2e,columns=cols_X)
X2E = pd.concat([Xtest_features_df,BB_predictions_df],axis=1)

dt_params = {
    'max_depth': [None, 10, 20, 30, 40, 50, 70, 80, 90, 100],
    'min_samples_split': [2**i for i in range(1, 10)],
    'min_samples_leaf': [2**i for i in range(1, 10)],
}
cv=5

logger.info('Going into the loop over all instances to explain')

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

    #logger.info('generating MIXED synthetic neighborhood')
    alpha_beta_sample_knn = synthetic_neighborhood.sample_alphaFeat_betaLabel(sampleKnn_feat_space,sampleKnn_label_space,alpha=0.7,beta=0.3)
    synthetic_neighborhood1 = synthetic_neighborhood.random_synthetic_neighborhood_df(
        alpha_beta_sample_knn,size,
        discrete_var=X2E[columns_type_dataset[dataset][1]].columns.values,
        continuous_var=X2E[columns_type_dataset[dataset][0]].columns.values,
        classes_name=cols_Y_BB)
    #using bb to predict syn instances output
    BB_label_syn1_df = pd.DataFrame(bb.predict(synthetic_neighborhood1.values),columns=cols_Y_BB)
    synthetic_neighborhood1 = pd.concat([synthetic_neighborhood1,BB_label_syn1_df],1)
    
    #####################################################################################################
    #growing tree on synthetic neighborhood
    #logger.info('growing DT on MIXED synthetic neighborhood')
    tree1 = DecisionTreeClassifier()
    sop = np.prod([len(v) for k, v in dt_params.items()])
    n_iter_search = min(100, sop)
    random_search = RandomizedSearchCV(tree1, param_distributions=dt_params,scoring='f1_micro', n_iter=n_iter_search, cv=cv)
    random_search.fit(synthetic_neighborhood1.drop(cols_Y_BB,1).values,synthetic_neighborhood1[cols_Y_BB].values)
    best_params = random_search.best_params_
    tree1.set_params(**best_params)
    tree1.fit(synthetic_neighborhood1.drop(cols_Y_BB,1).values,synthetic_neighborhood1[cols_Y_BB].values)
    ## evuluating on synthetic neighborhood
    y_syn1 = synthetic_neighborhood1[cols_Y_BB].values
    y_tree1_syn1 = tree1.predict(synthetic_neighborhood1.drop(cols_Y_BB,1).values)
    f1_syn1_tree1 = f1_score(y_true=y_syn1,y_pred=y_tree1_syn1, average='micro')
    ## evuluating on real kNN neighborhood used to create the synthetic one
    y_samplekNN1 = bb.predict(alpha_beta_sample_knn.drop(cols_Y_BB,1).values)
    y_tree1_kNN1 = tree1.predict(alpha_beta_sample_knn.drop(cols_Y_BB,1).values)
    f1_kNN1_tree1 = f1_score(y_true=y_samplekNN1,y_pred=y_tree1_kNN1, average='micro')
    #hit
    y_hit_tree1 = tree1.predict(i2e_values.reshape(1, -1))
    hit_jc_tree1 = jaccard_similarity_score(y_i2e_bb,y_hit_tree1)
    hit_sm_tree1 = distance_functions.simple_match_distance(y_i2e_bb[0],y_hit_tree1[0])
    #rule tree1
    rule_tree1,len_rule1 = rules.istance_rule_extractor(i2e_values.reshape(1, -1),tree1,cols_X)

    #####################################################################################################
    #################################UNIFIED NEIGHBORHOOD################################################

    
    #logger.info('generating UNIFIED synthetic neighborhood')
    synthetic_neighborhood2 = synthetic_neighborhood.random_synthetic_neighborhood_df(
        sampleKnn_mixed_space,size,discrete_var=X2E[columns_type_dataset[dataset][1]].columns.values,
        continuous_var=X2E[columns_type_dataset[dataset][0]].columns.values, classes_name=cols_Y_BB)
    #using bb to predict syn instances output
    BB_label_syn2_df = pd.DataFrame(bb.predict(synthetic_neighborhood2.values),columns=cols_Y_BB)
    synthetic_neighborhood2 = pd.concat([synthetic_neighborhood2,BB_label_syn2_df],1)
    #####################################################################################################
    #growing tree on synthetic neighborhood        
    #logger.info('growing DT on UNIFIED synthetic neighborhood')
    tree2 = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(tree2, param_distributions=dt_params,scoring='f1_micro', n_iter=n_iter_search, cv=cv)
    random_search.fit(synthetic_neighborhood2.drop(cols_Y_BB,1).values,synthetic_neighborhood2[cols_Y_BB].values)
    best_params = random_search.best_params_
    tree2.set_params(**best_params)
    tree2.fit(synthetic_neighborhood2.drop(cols_Y_BB,1).values,synthetic_neighborhood2[cols_Y_BB].values)
    #print('____EVALUATION TREE2___bb: %s' % blackbox_name)
    ## evuluating on synthetic neighborhood
    y_syn2 = synthetic_neighborhood2[cols_Y_BB].values
    y_tree2_syn2 = tree1.predict(synthetic_neighborhood2.drop(cols_Y_BB,1).values)
    f1_syn2_tree2 = f1_score(y_true=y_syn2,y_pred=y_tree2_syn2, average='micro')
    ## evuluating on real kNN neighborhood used to create the synthetic one
    y_samplekNN2 = bb.predict(sampleKnn_mixed_space.drop(cols_Y_BB,1).values)
    y_tree2_kNN2 = tree2.predict(sampleKnn_mixed_space.drop(cols_Y_BB,1).values)
    f1_kNN2_tree2 = f1_score(y_true=y_samplekNN2,y_pred=y_tree2_kNN2, average='micro')
    ## hit
    y_hit_tree2 = tree2.predict(i2e_values.reshape(1, -1))
    hit_jc_tree2 = jaccard_similarity_score(y_i2e_bb,y_hit_tree2)
    hit_sm_tree2 = distance_functions.simple_match_distance(y_i2e_bb[0],y_hit_tree2[0])
    #rule tree2
    rule_tree2,len_rule2 = rules.istance_rule_extractor(i2e_values.reshape(1, -1),tree2,cols_X)


    jrow = {
        'dataset_name':dataset,
        'bb_name': blackbox_name,
        'i2e_id': str(instance),
        'fidelity_tree1_syn':f1_syn1_tree1,
        'fidelity_tree1_kNN':f1_kNN1_tree1,
        'fidelity_tree2_syn':f1_syn2_tree2,
        'fidelity_tree2_kNN':f1_kNN2_tree2,
        'i2e_bb_label': y_i2e_bb[0].astype(int).tolist(),
        'i2e_tree1_label':y_hit_tree1[0].astype(int).tolist(),
        'i2e_tree2_label':y_hit_tree2[0].astype(int).tolist(),
        'hit_sm_tree1':hit_sm_tree1,
        'hit_jc_tree1':hit_jc_tree1,
        'hit_sm_tree2':hit_sm_tree2,
        'hit_jc_tree2':hit_jc_tree2,
        'rule_tree1':rule_tree1,
        'rule_tree2':rule_tree2,
        'lenght_rule_tree1':len_rule1,
        'lenght_rule_tree2':len_rule2,
    }
    
    try:
        json_str = ('%s\n' % json.dumps(jrow)).encode('utf-8')
    except Exception:
        logger.exception('Problems in dumping row')
        break
    try:
        with gzip.GzipFile('../output/%s_%s_%s_explanationsandmetrics.json.gz' % (now, dataset, blackbox_name), 'a') as fout:
            fout.write(json_str)
    except Exception:
        logger.exception('Problems in saving the output')
        break 

logging.info('end')
