import pandas as pd
import numpy as np


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
