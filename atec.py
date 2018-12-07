import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import catboost as cat
import xgboost as xgb

path = '../../data/atec_anti_fraud_train.csv'
data = pd.read_csv(path)

data = data[data['label']!=-1]
data = data.drop(columns=['f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47'])

#
feat_name = [ 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 
             'f15', 'f16', 'f17', 'f18', 'f19',
             'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62','f63', 'f72', 'f73', 'f74', 'f80', 'f88', 'f89', 'f90',
             'f103', 'f104', 'f109', 'f110', 'f157', 'f158', 'f159', 'f166',
             'f167', 'f168', 'f169', 'f170', 'f171',
             'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178', 'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 
             'f185', 'f186', 'f187', 'f188',
             'f194', 'f195',  
             'f199', 'f200', 'f204', 'f205', 'f206', 'f207', 'f208', 'f209', 'f210', 'f211', 'f212', 
             'f213', 'f214', 'f215', 'f216', 'f217', 'f218', 'f219', 'f220', 'f221', 'f222', 'f223', 'f224', 'f225', 'f226', 
             'f227', 'f228', 'f229', 'f230', 'f231', 'f232', 'f233', 'f234', 'f235', 'f236', 'f237', 'f238', 'f239', 'f240', 
             'f241', 'f242', 'f243', 'f244', 'f245', 'f246', 'f247', 'f248', 'f249', 'f250', 'f251', 'f252', 'f253', 'f254', 
             'f255', 'f256', 'f259', 'f260', 'f261', 'f262', 'f263', 'f264', 'f265', 'f266', 'f267', 
             'f269', 'f270', 'f271', 'f272', 'f273',  'f275', 'f276', 
             'f278', 'f279', 'f280',
             'f288', 'f289', 'f290', 'f291', 'f292',
             ] 
			 
train1 = data[(data['date']>=20170905)&(data['date']<20170917)]
train2 = data[(data['date']>=20170917)&(data['date']<20170929)]
train3 = data[(data['date']>=20170929)&(data['date']<20171011)]
train4 = data[(data['date']>=20171011)&(data['date']<20171023)]
train5 = data[(data['date']>=20171023)&(data['date']<=20171105)]

gbm1 = lgb.LGBMClassifier(max_depth=7, n_estimators = 110,  min_child_samples=100)
gbm2 = lgb.LGBMClassifier(max_depth=7, n_estimators = 110,  min_child_samples=100)
gbm3 = lgb.LGBMClassifier(max_depth=7, n_estimators = 110,  min_child_samples=100)
gbm4 = lgb.LGBMClassifier(max_depth=7, n_estimators = 110,  min_child_samples=100)
gbm5 = lgb.LGBMClassifier(max_depth=7, n_estimators = 110,  min_child_samples=100)
gbm1.fit(train1[feat_name], train1['label'])
gbm2.fit(train2[feat_name], train2['label'])
gbm3.fit(train3[feat_name], train3['label'])
gbm4.fit( train4[feat_name], train4['label'])
gbm5.fit( train5[feat_name],  train5['label'])

cb1 = cat.CatBoostClassifier(iterations=110,learning_rate=0.1,depth=7)
cb2 = cat.CatBoostClassifier(iterations=110,learning_rate=0.1,depth=7)
cb3 = cat.CatBoostClassifier(iterations=110,learning_rate=0.1,depth=7)
cb4 = cat.CatBoostClassifier(iterations=110,learning_rate=0.1,depth=7)
cb5 = cat.CatBoostClassifier(iterations=110,learning_rate=0.1,depth=7)
cb1.fit(train1[feat_name], train1['label'],verbose=20)
cb2.fit(train2[feat_name], train2['label'],verbose=20)
cb3.fit(train3[feat_name], train3['label'],verbose=20)
cb4.fit(train4[feat_name], train4['label'],verbose=20)
cb5.fit(train5[feat_name], train5['label'],verbose=20)

xg1 = xgb.XGBClassifier(max_depth=7, n_estimators=110, silent=False)
xg2 = xgb.XGBClassifier(max_depth=7, n_estimators=110, silent=False)
xg3 = xgb.XGBClassifier(max_depth=7, n_estimators=110, silent=False)
xg4 = xgb.XGBClassifier(max_depth=7, n_estimators=110, silent=False)
xg5 = xgb.XGBClassifier(max_depth=7, n_estimators=110, silent=False)
xg1.fit(train1[feat_name], train1['label'], verbose=20)
xg2.fit(train2[feat_name], train2['label'], verbose=20)
xg3.fit(train3[feat_name], train3['label'], verbose=20)
xg4.fit(train4[feat_name], train4['label'], verbose=20)
xg5.fit(train5[feat_name], train5['label'], verbose=20)

test = pd.read_csv('../../data/atec_anti_fraud_test_b.csv')
y_pred1 = xg1.predict_proba(test[feat_name])[:, 1]
y_pred2 = xg2.predict_proba(test[feat_name])[:, 1]
y_pred3 = xg3.predict_proba(test[feat_name])[:, 1]
y_pred4 = xg4.predict_proba(test[feat_name])[:, 1]
y_pred5 = xg5.predict_proba(test[feat_name])[:, 1]
y_pred_xgb = (y_pred1+y_pred2+y_pred3+y_pred4+y_pred5)/5

y_pred1 = cb1.predict_proba(test[feat_name])[:, 1]
y_pred2 = cb2.predict_proba(test[feat_name])[:, 1]
y_pred3 = cb3.predict_proba(test[feat_name])[:, 1]
y_pred4 = cb4.predict_proba(test[feat_name])[:, 1]
y_pred5 = cb5.predict_proba(test[feat_name])[:, 1]
y_pred_cb = (y_pred1+y_pred2+y_pred3+y_pred4+y_pred5)/5

y_pred1 = gbm1.predict_proba(test[feat_name])[:, 1]
y_pred2 = gbm2.predict_proba(test[feat_name])[:, 1]
y_pred3 = gbm3.predict_proba(test[feat_name])[:, 1]
y_pred4 = gbm4.predict_proba(test[feat_name])[:, 1]
y_pred5 = gbm5.predict_proba(test[feat_name])[:, 1]
y_pred_gbm = (y_pred1+y_pred2+y_pred3+y_pred4+y_pred5)/5

ans = pd.DataFrame()
ans['id'] = test['id']
ans['score'] = 0.33*y_pred_gbm + 0.33*y_pred_xgb + 0.34*y_pred_cb
ans.to_csv('../../rh.csv', index=False)
