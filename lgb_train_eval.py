# coding=utf-8

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import metrics
import bisect

train1=pd.read_csv('../../output/latest/train1.csv')
train2=pd.read_csv('../../output/latest/train2.csv')
print(train1.shape,train2.shape)

train1_y = train1.label
train1_x = train1.drop('label',axis=1)
colu=list(train1_x.columns)
del train1
lgb_train1 = lgb.Dataset(train1_x,train1_y)
del train1_x,train1_y

train2_y = train2.label
train2_x = train2.drop('label',axis=1)
del train2
lgb_train2 = lgb.Dataset(train2_x,train2_y)
del train2_x,train2_y






def get_tpr_from_fpr(fpr_array, tpr_array, target):
    fpr_index = np.where(fpr_array == target)
    assert target <= 0.01, 'the value of fpr in the custom metric function need lt 0.01'
    if len(fpr_index[0]) > 0:
        return np.mean(tpr_array[fpr_index])
    else:
        tmp_index = bisect.bisect(fpr_array, target)
        fpr_tmp_1 = fpr_array[tmp_index-1]
        fpr_tmp_2 = fpr_array[tmp_index]
        if (target - fpr_tmp_1) > (fpr_tmp_2 - target):
            tpr_index = tmp_index
        else:
            tpr_index = tmp_index - 1
        return tpr_array[tpr_index]


def eval_metric(pred, labels):
    labels=labels.get_label()
    fpr, tpr, _ = metrics.roc_curve(labels, pred, pos_label=1)
    tpr1 = get_tpr_from_fpr(fpr, tpr, 0.001)
    tpr2 = get_tpr_from_fpr(fpr, tpr, 0.005)
    tpr3 = get_tpr_from_fpr(fpr, tpr, 0.01)
    return 'score',0.4*tpr1 + 0.3*tpr2 + 0.3*tpr3,True

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'lambda_l1': 2.0,
    'lambda_l2': 5.0,
    'min_gain_to_split': 0.001,
    'min_sum_hessian_in_leaf': 1.0,
    'max_depth': 8,
    'bagging_freq': 1
}

'''



'''
gbm = lgb.train(params,
                lgb_train1,
                num_boost_round=20000,
                valid_sets=[lgb_train1, lgb_train2],

                feval=eval_metric,
                early_stopping_rounds=100,
                categorical_feature=colu
                )



# save feature score
fs = pd.DataFrame(columns=['feature', 'score'])
fs['feature'] = list(gbm.feature_name())
fs['score'] = list(gbm.feature_importance())
fs.sort_values(by='score', ascending=False, inplace=True)
fs.to_csv('../../result/lgb_feature_score.csv', index=None)

