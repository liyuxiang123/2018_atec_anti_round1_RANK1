import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb


data_train=pd.read_csv('../input/atec_anti_fraud_train.csv')





data_train = data_train.drop(['id'], axis=1).sort_values(by='date')

colms = list(data_train.columns)[2:]
feat_drop = []
for colu in colms:
    if data_train[colu].isnull().sum() > 200000:
        feat_drop.append(colu)

data_train=data_train.fillna(-1)
data_train.label=data_train.label.replace(-1,1)

select_feat=['f5','f20','f28','f32','f36','f48','f52','f54','f64','f72','f76','f102','f107','f111','f155','f161','f166','f211','f254','f278']

def produce_feat(dataset):
    for feat in select_feat:
        dataset[feat+'solve_nan']=dataset[feat].apply(lambda x:x if x==-1 else 1)
    return dataset

train=produce_feat(data_train)
train=train.drop(feat_drop,axis=1)
train.drop('date',axis=1,inplace=True)
del data_train


data_test = pd.read_csv('../input/atec_anti_fraud_test_b.csv')
data_test=data_test.fillna(-1)
test=produce_feat(data_test)
del data_test
test.drop(feat_drop,axis=1,inplace=True)
test=test.drop('date',axis=1)


def preprocess(data: pd.DataFrame):
    columns = data.columns
    for col_name in columns:
        mode = data[col_name].mode().values[0]
        data[col_name] = data[col_name].fillna(mode).astype('float64')

    return data


train_col=list(train.columns)[2:]
test_col=list(test.columns)[2:]
train[train_col]=preprocess(train[train_col].copy())
test[test_col]=preprocess(test[test_col].copy())


num_train=40

train_data=[]
for i in range(num_train):
    train_data.append(train.iloc[i*25000:i*25000+25000])



col1 = list(train.columns)[2:]

lg=[]
xgbmodel=[]
res=[]
grad=[]
rf=[]

for i in range(num_train):

    lg.append(lgb.LGBMClassifier(max_depth = 8, n_estimators = 100,num_leaves=31))
    xgbmodel.append(xgb.XGBClassifier(max_depth = 8, n_estimators = 30,num_leaves=31))
    grad.append(GradientBoostingClassifier(n_estimators=15,max_depth=10))


clf = []
for i in range(num_train):
      clf.append(VotingClassifier(estimators = [('lg',lg[i]), ('xgb',xgbmodel[i]),('grad',grad[i])], voting = 'soft'))
      clf[i].fit(train_data[i][col1].as_matrix(),train_data[i]['label'])
      res.append(clf[i].predict_proba(test[col1].as_matrix())[:,1])



end=pd.DataFrame({'id':test['id'],'score':sum(res)/40})
print(end.describe())
end.to_csv('./0707_1.csv',index=False)
