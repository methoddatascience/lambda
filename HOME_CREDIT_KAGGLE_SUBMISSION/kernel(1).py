#HOME_CREDIT_DEFAULT_RISK submitted by team Lambda 

import pandas as pd
import numpy as np


from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


import gc



dataset = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')

#Separate target variable
y = dataset['TARGET']
del dataset['TARGET']

#One-hot encoding of categorical features in data and test sets
categorical_features = [col for col in dataset.columns if dataset[col].dtype == 'object']

one_hot_df = pd.concat([dataset,test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)

dataset = one_hot_df.iloc[:dataset.shape[0],:]
test = one_hot_df.iloc[dataset.shape[0]:,]


#delete features with too many missing data
test = test[test.columns[dataset.isnull().mean() < 0.80]]
dataset = dataset[dataset.columns[dataset.isnull().mean() < 0.80]]

from lightgbm import LGBMClassifier
import gc

gc.enable()

folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(dataset.shape[0])
test_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()

ftr = [f for f in dataset.columns if f not in ['SK_ID_CURR']]

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(dataset)):
    trn_x, trn_y = dataset[ftr].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = dataset[ftr].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=34,
        colsample_bytree=0.9,
        subsample=0.8,
        max_depth=8,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=250,
        silent=-1,
        verbose=-1,
        )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    test_preds += clf.predict_proba(test[ftr], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = ftr
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.5f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 

test['TARGET'] = test_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('submission_2_lambda.csv', index=False)




