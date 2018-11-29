import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from multilabelexplanations import rules


def sample_alphaFeat_betaLabel(samplekNN_feat, samplekNN_label, alpha=0.7,beta=0.3):
    
    """
    samplekNN_feat: pandas dataframe, it contains the features from the k nearest neighbors of the instance to be explained in the feature space 
    samplekNN_label: pandas dataframe, it contains the features from the k nearest neighbors of the instance to be explained in the label space 
    alpha: double, default 0.7, % of samples to be taken from samplekNN_feat
    beta: double, default 0.3, % of samples to be taken from samplekNN_label
    
    This function takes two sample of X2E instances taken according to:
            * their distance from the instance to be explained (i2e) in the FEATURE SPACE; samplekNN_feat
            * their distance from the instance to be explained (i2e) in the LABEL SPACE; samplekNN_label
    """
   
    if alpha+beta==1.:
        subsample_knn_feat = samplekNN_feat.sample(frac=alpha).reset_index().drop('index',1)
        subsample_knn_label = samplekNN_label.sample(frac=beta).reset_index().drop('index',1)
        alpha_beta_sample_knn = pd.concat([subsample_knn_feat,subsample_knn_label]).reset_index().drop('index',1)
    
        if len(alpha_beta_sample_knn)<len(samplekNN_feat):
            n = len(samplekNN_feat)-len(alpha_beta_sample_knn) 
            if alpha > beta:
                alpha_beta_sample_knn = pd.concat([alpha_beta_sample_knn,samplekNN_feat.sample(n=n).reset_index().drop('index',1)])
            else:
                alpha_beta_sample_knn = pd.concat([alpha_beta_sample_knn,samplekNN_label.sample(n=n).reset_index().drop('index',1)])
        
            alpha_beta_sample_knn = alpha_beta_sample_knn.reset_index().drop('index',1)
    else:
        print('ERROR: "alpha + beta" must be = 1')
        return
        
    return alpha_beta_sample_knn


def random_synthetic_neighborhood_df(sample_Knn, size, discrete_var, continuous_var, classes_name):
    """This function takes as input:
            sample_Knn: dataframe, K nearest neighbors of the instance to be explained
            size: int, the number of synthetic instances to generate
            discrete_var: list, name of columns containing discrete variables
            continuous_var: list, name of columns containing continuos variables
            classes_name: list, name of columns containing the classes labels
        And it generates a synthetic neighbothood of instances sampling from features distributions of the sample of K
        nearest neighbors given
    """
    df = sample_Knn.drop(classes_name,1)
    
    if len(continuous_var)>0:
        #print('there are continuos variables')
        cont_cols_synthetic_instances = list()
        for col in continuous_var:
            values = df[col].values
            mu = np.mean(values)
            sigma = np.std(values)
            new_values = np.random.normal(mu,sigma,size)
            cont_cols_synthetic_instances.append(new_values)
        
        cont_col_syn_df = pd.DataFrame(data=np.column_stack(cont_cols_synthetic_instances),columns=continuous_var)
    
    if len(discrete_var)>0:
        #print('there are discrete variables')
        disc_cols_synthetic_instances = list()
        for col in discrete_var:
            values = df[col].values
            diff_values = np.unique(values)
            prob_values = [1.0 * list(values).count(val) / len(values) for val in diff_values]
            new_values = np.random.choice(diff_values, size, prob_values)
            disc_cols_synthetic_instances.append(new_values)
        
        disc_col_syn_df = pd.DataFrame(data=np.column_stack(disc_cols_synthetic_instances),columns=discrete_var)
    
    if (len(continuous_var)>0)&(len(discrete_var)>0): 
        return pd.concat([cont_col_syn_df,disc_col_syn_df],axis=1)
    
    elif len(continuous_var)==0:
        #print('there are no continuous variables')
        return disc_col_syn_df 
    
    elif len(discrete_var)==0:
        #print('there are no discrete variables')
        return cont_col_syn_df
    else:
        print('Error, no variables in input df')


def tuned_tree(X,y,param_distributions,scoring='f1_micro',cv=5):
    """
    This function takes as input a traning set 
    - X: array-like or sparse matrix, shape = [n_samples, n_features], the training input sample
    - y: array-like, shape = [n_samples] or [n_samples, n_outputs], the target values (class labels) as integers or strings.
    - param_distributions: dict, dictionary with parameters names (string) as keys and distributions or lists of parameters to try. 
    - scoring : string, callable, list/tuple, dict or None, default 'f1_micro'
    - cv: int (number of folds), cross-validation generator or an iterable, optional, it determines the cross-validation splitting strategy, default=5

    performs an hyperpatameter tuning using a randomized search (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    and returns a tuned decision tree
    """

    tree = DecisionTreeClassifier()
    sop = np.prod([len(v) for k, v in param_distributions.items()])
    n_iter_search = min(100, sop)
    random_search = RandomizedSearchCV(tree, param_distributions=param_distributions,scoring='f1_micro', n_iter=n_iter_search, cv=cv)
    random_search.fit(X, y)
    best_params = random_search.best_params_
    tree.set_params(**best_params)
    return tree

def oneDTforlabelpreds(i2e, synthetic_neigh, knn_neigh, cols_X, cols_Y_BB, param_distributions):
    """
    This function takes:
    - i2e: pd.Series, the instance to be explained
    - synthetic_neigh: pd.DataFrame, the synthetic neighborhood of i2e
    - knn_neigh: pd.DataFrame, the core of real neighbors of i2e
    - cols_X: list, names of the columns containing the features
    - cols_Y_BB: list, names of the columns containing the labels
    - param_distributions: dict, dictionary with parameters names (string) as keys and distributions or lists of parameters to try for the hyperparameter tuning of the DTs

    and computes a DT that is trained to learn one of the labels, than it performs a concatenations of all the predictions
    and returns:
    
    - unionDT_labels_syn: pd.DataFrame
    - unionDT_labels_knn: pd.DataFrame
    - unionDT_labels_i2e: pd.Series
    - DT_rules: list, rules, one for each label
    - DT_rules_len: list, lenght of rules
    
    """
    DT_syn_labelspred = {}
    DT_knn_labelspred = {}
    DT_i2e_labelspred = {}
    DT_rules = []
    DT_rules_len = []
    size = len(synthetic_neigh)
    
    for label in synthetic_neigh[cols_Y_BB].columns.values:
        label_col = synthetic_neigh[label]
        not_dummy = len(label_col.drop_duplicates())>1
        imbalanced = label_col.value_counts(normalize=True).iloc[0]>0.8
        columns_DT = np.append(cols_X,label)
        y_i2e_bb_label = i2e[label]
        
        #if not all in one class (0 or 1)
        if not_dummy:
            X = synthetic_neigh[cols_X].values
            y = synthetic_neigh[label].values

            #if the % of instances belonging to the majority class is >80% 
            if imbalanced:
                #oversampling with 'auto' (not majority) sampling strategy
                ros = RandomOverSampler(sampling_strategy='auto',random_state=0)
                X_resampled, y_resampled = ros.fit_resample(X, y)
                #it creates a balanced (50/50) syn neigh with set size (1000)
                set_size_syn_neigh = pd.DataFrame(np.c_[X_resampled, y_resampled],columns=columns_DT).groupby(label, group_keys=False).apply(lambda x: x.sample(int(size/2),random_state=0)).reset_index().drop('index',1)
                X_resampled = set_size_syn_neigh[cols_X].values
                y_resampled = set_size_syn_neigh[label].values

                #fine tuning of the DT
                tree = tuned_tree(X_resampled,y_resampled,param_distributions=param_distributions)
                #training the DT
                tree.fit(X_resampled, y_resampled)

            else:
                #not imbalanced
                #fine tuning of the DT
                tree = tuned_tree(X,y,param_distributions=param_distributions)
                #training the DT
                tree.fit(X, y)

            ########################################
            #DT prediction on the synthetic neighborhood (y_tree1_syn1)
            DT_syn_labelspred[label]= tree.predict(X)
            #DT prediction on the real neighborhood (y_tree1_kNN1)
            y_samplekNN = knn_neigh[cols_Y_BB]

            y_samplekNN = knn_neigh[label].values
            DT_knn_labelspred[label] = tree.predict(knn_neigh[cols_X].values)
            ########################################

            #rule tree1
            rule_tree,len_rule = rules.istance_rule_extractor(i2e[cols_X].values.reshape(1, -1),tree,cols_X)
            rule_tree = rule_tree + ' ('+label+')'
            DT_rules.append(rule_tree)
            DT_rules_len.append(len_rule)
            #prediction of i2e
            DT_i2e_labelspred[label] = tree.predict(i2e[cols_X].values.reshape(1, -1))

        else: #if the column is dummy the tree will be dummy, I'm not growing a tree, it's a waste of time
            #print('%s is dummy:' %label)
            X = synthetic_neigh[cols_X].values
            y = synthetic_neigh[label].values
            rule_tree = '->['+str(int(y[0]))+']'+' ('+label+')'
            len_rule = 0

            DT_rules.append(rule_tree)
            DT_rules_len.append(len_rule)
            DT_syn_labelspred[label]=y
            DT_knn_labelspred[label]=y[0:len(knn_neigh)]
            #prediction of i2e
            DT_i2e_labelspred[label] = y[0]
        
    unionDT_labels_syn = pd.DataFrame(DT_syn_labelspred)
    unionDT_labels_knn = pd.DataFrame(DT_knn_labelspred)
    unionDT_labels_i2e = pd.DataFrame(DT_i2e_labelspred)        
    
    return unionDT_labels_syn, unionDT_labels_knn, unionDT_labels_i2e, DT_rules, DT_rules_len     
