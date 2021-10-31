import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42

SEED = 42

PATH = '.'
#PATH = '/kaggle/input'

data_p1_link = PATH+'/risk-management-uiim/clean_train_part1.pkl'
data_p2_link = PATH+'/risk-management-uiim/clean_train_part2.pkl'
data_test_link = PATH+'/risk-management-uiim/test_data.pkl'
submission_link = PATH+'/risk-management-uiim/submission.csv'


data_p1 = pd.read_pickle(data_p1_link)
data_p2 = pd.read_pickle(data_p2_link)
data_train = data_p1.append(data_p2)
del data_p1
del data_p2
#data_train.dropna(thresh=50, inplace=True)
data_train.fillna(method='ffill', inplace=True)
#data_train.drop([['x_618', 'x_617', 'x_17', 'x_615', 'x_25', 'x_26', 'x_27']], axis=1, inplace=True)


from catboost import CatBoostClassifier, Pool


FEATURES = [col for col in data_train.columns if col != 'TARGET']


def preprocess(data, FEATURES):
    #find number of years the loan was issued 
    data['years'] = data['REPORT_DT'].dt.year - data['x_9'].dt.year    
    FEATURES += ['years']

    #x_19, x_634 and x_614 need to be converted to int
    for col in ['x_19', 'x_614', 'x_634', 'x_13']:
        #x_13 too after processing None
        if col == 'x_13':
            data['x_13'] = data['x_13'].fillna(0)
        data[col] = data[col].astype('int8')
        FEATURES += [col]

    #ordinal_encode credit ratings x_12
    data['x_12'] = data['x_12'].apply(lambda x: ['N', 'D', 'C', 'B', 'B1', 'A', 'A1'].index(x))
    FEATURES += ['x_12']

    #one hot encoding some cols
#    for col in ['x_18', 'x_21', 'x_625', 'x_628']:
#        dummies = pd.get_dummies(data[col], prefix=col)
#        data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
#        FEATURES += list(dummies.columns)

    #update FEATURES
    FEATURES = list(set(FEATURES))
    FEATURES.sort()

    return data, FEATURES

data_train, FEATURES = preprocess(data_train, FEATURES)



train_data = Pool(data=data_train[FEATURES],
                  label=data_train['TARGET'],cat_features=['x_12', 'x_13', 'x_18', 'x_19', 'x_21', 'x_614', 'x_625', 'x_628','x_634'])

model = CatBoostClassifier(iterations=450,learning_rate=0.01)

model.fit(train_data)
preds_class = model.predict_proba(train_data)


data_test = pd.read_pickle(data_test_link)
data_test.fillna(method='ffill', inplace=True)
#data_test.drop([['x_618', 'x_617', 'x_17', 'x_615', 'x_25', 'x_26', 'x_27']], axis=1, inplace=True)

data_test, FEATURES = preprocess(data_test, FEATURES)

submission = pd.read_csv(submission_link)
submission['Probability'] = model.predict_proba(data_test[FEATURES])[:,1]
submission.to_csv('submission_cat.csv',index=False)

X_train, X_test, y_train, y_test = train_test_split(data_train[FEATURES], data_train['TARGET'], test_size=0.33, random_state=SEED)


print('testing')
print('TRAIN SCORE', roc_auc_score(y_train,model.predict_proba(X_train)[:,1]))
print('TEST SCORE', roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))


# Submit
#subprocess.run('kaggle competitions submit -c risk-management-uiim -f submission_cat.csv -m "Message"', shell=True)
# View results
#subprocess.run('kaggle competitions submissions -c risk-management-uiim', shell=True)
