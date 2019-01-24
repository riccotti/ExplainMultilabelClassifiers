import pickle

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skmultilearn.dataset import load_from_arff


#dictionaries with key: dataset, value: list of lists, 0: continuos variables, 1: discrete variables
columns_type_dataset = {'yeast':[['Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7', 'Att8', 'Att9', 'Att10',\
                                  'Att11','Att12', 'Att13','Att14','Att15','Att16','Att17','Att18','Att19','Att20',\
                                  'Att21', 'Att22', 'Att23', 'Att24', 'Att25', 'Att26', 'Att27', 'Att28', 'Att29', \
                                  'Att30', 'Att31', 'Att32', 'Att33', 'Att34', 'Att35', 'Att36', 'Att37', 'Att38',\
                                  'Att39', 'Att40','Att41','Att42', 'Att43', 'Att44', 'Att45', 'Att46', 'Att47',\
                                  'Att48', 'Att49', 'Att50', 'Att51', 'Att52', 'Att53', 'Att54', 'Att55', 'Att56',\
                                  'Att57', 'Att58', 'Att59', 'Att60', 'Att61', 'Att62','Att63', 'Att64', 'Att65',\
                                  'Att66', 'Att67', 'Att68', 'Att69', 'Att70', 'Att71', 'Att72', 'Att73', 'Att74', \
                                  'Att75', 'Att76', 'Att77', 'Att78', 'Att79', 'Att80', 'Att81', 'Att82', 'Att83',\
                                  'Att84','Att85', 'Att86', 'Att87', 'Att88', 'Att89', 'Att90', 'Att91', 'Att92',\
                                  'Att93', 'Att94', 'Att95', 'Att96', 'Att97', 'Att98', 'Att99', 'Att100', 'Att101',\
                                  'Att102', 'Att103'],[]],\
                        'diabete':[['age','time_in_hospital','num_lab_procedures','num_procedures','num_medications',\
                                    'number_outpatient','number_emergency','number_inpatient','number_diagnoses'],\
                                   ['race','gender','admission_type_id','discharge_disposition_id',\
                                    'admission_source_id','max_glu_serum','A1Cresult','diabetesMed',\
                                    'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',\
                                    'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',\
                                    'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide',\
                                    'citoglipton','insulin','glyburide-metformin','glipizide-metformin',\
                                    'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone',\
                                    'change','readmitted','diag_1','diag_2','diag_3']],\
                        'medical':[[],[]],\
                        'woman':[['n_0067','n_0078','n_0108','n_0109','o_0176','o_0264'],\
                                 ['release','n_0047','n_0050','n_0052','n_0061','n_0075','n_0091','c_0466',\
                                  'c_0500','c_0638','c_0699','c_0738','c_0761','c_0770','c_0838','c_0870',\
                                  'c_0980','c_1145','c_1158','c_1189','c_1223','c_1227','c_1244','c_1259']]}


########### PREPARE MEDICAL DATASET
medicals_dataset = ['medical-test','medical-train']

for dataset in medicals_dataset:
    data = load_from_arff('../dataset/dataset_raw/%s.arff' % dataset, \
                          label_count=45, \
                          load_sparse=False, \
                          return_attribute_definitions=True)
    cols_X = [i[0] for i in data[2]]
    cols_Y = [i[0] for i in data[3]]
    print('dataset %s, cols X: %d, cols Y: %d' %(dataset,len(cols_X),len(cols_Y)))
    print('n_instances in dataset %s: %d' %(dataset,len(data[0].todense())))
    
    X_med_df = pd.DataFrame(data[0].todense(),columns=cols_X)
    y_med_df = pd.DataFrame(data[1].todense(),columns=cols_Y)
    
    medical_df = pd.concat([X_med_df,y_med_df],1)
    
    if dataset == 'medical-test':
        medical_df.to_csv('../dataset/medical_2e.csv', sep=',', index=False)
        print('medical_2e.csv')
    else:
        medical_df.to_csv('../dataset/medical_bb.csv', sep=',', index=False)
        print('medical_bb.csv')
        
columns_type_dataset['medical'][1] = cols_X



########### PREPARE YEAST DATASET
df_yeast = pd.DataFrame(arff.loadarff('../dataset/dataset_raw/yeast.arff')[0])

for col in df_yeast.columns[-14:]:
    df_yeast[col] = df_yeast[col].apply(pd.to_numeric)

cols_Y = [col for col in df_yeast.columns if col.startswith('Class')]
cols_X = [col for col in df_yeast.columns if col not in cols_Y]

X = df_yeast[cols_X].values
y = df_yeast[cols_Y].values

X_bb, X_2e, y_bb, y_2e = train_test_split(X, y, test_size=0.3, random_state=0)


df_bb = pd.DataFrame(data=np.concatenate((X_bb, y_bb), axis=1), columns=df_yeast.columns)
df_2e = pd.DataFrame(data=np.concatenate((X_2e, y_2e), axis=1), columns=df_yeast.columns)


df_bb.to_csv('../dataset/yeast_bb.csv', sep=',', index=False)
df_2e.to_csv('../dataset/yeast_2e.csv', sep=',', index=False)
print('yeast_bb.csv')
print('yeast_2e.csv')


######## PREPARE WOMAN DATASET
woman = pd.read_csv('../dataset/dataset_raw/women_health_care.csv', sep=',')

mv = woman.isnull().sum(axis=0)

columns2drop = list()
for k, v in zip(mv.index, mv.values):
    if v != 0.0:
        columns2drop.append(k)

woman.drop(columns2drop, axis=1, inplace=True)

enc = OneHotEncoder(categories='auto')
encoded_dataframes = []
categorical_variables_woman = []

for col in columns_type_dataset['woman'][1]:
    enc.fit(woman[col].values.reshape(-1,1))
    categories_names = enc.categories_
    columns_names = [col+str(name) for name in categories_names[0]]
    values_encoded = enc.transform(woman[col].values.reshape(-1, 1)).toarray()
    encoded_dataframes.append(pd.DataFrame(values_encoded, columns=columns_names))
    categorical_variables_woman.append(columns_names)

categorical_variables_woman_names = [item for sublist in categorical_variables_woman for item in sublist]
woman_encoded = pd.concat([pd.concat(encoded_dataframes,1),woman.drop(columns_type_dataset['woman'][1],1)],1)

columns_type_dataset['woman'][1] = categorical_variables_woman_names
df_hc_label = pd.read_csv('../dataset/dataset_raw/women_health_care_labels.csv', sep=',')

df_hc = woman_encoded.set_index('id').join(df_hc_label.set_index('id'), how='inner').reset_index().drop('id',1)

cols_Y = [col for col in df_hc.columns if col.startswith('service')]
cols_X = [col for col in df_hc.columns if col not in cols_Y]

X = df_hc[cols_X].values
y = df_hc[cols_Y].values

X_bb, X_2e, y_bb, y_2e = train_test_split(X, y, test_size=0.3, random_state=0)

df_bb = pd.DataFrame(data=np.concatenate((X_bb, y_bb), axis=1), columns=df_hc.columns)
df_2e = pd.DataFrame(data=np.concatenate((X_2e, y_2e), axis=1), columns=df_hc.columns)

df_bb.to_csv('../dataset/woman_bb.csv', sep=',', index=False)
df_2e.to_csv('../dataset/woman_2e.csv', sep=',', index=False)

print('woman_bb.csv')
print('woman_2e.csv')


#Save dictionary of varibles names divided in categorical and continuous ones for later use
with open('../dataset/dict_names.pickle', 'wb') as handle:
    pickle.dump(columns_type_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)