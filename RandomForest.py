import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix


train_path = "/content/drive/MyDrive/COMP0036CW/dataset/final_train_data.csv"
training_data = pd.read_csv(train_path)

X_train = training_data.drop(['FTR'], 1)
y_train = training_data['FTR']

'''
### encode data ###
def one_hot_encode(data, feature):
    dummies = pd.get_dummies(data[[feature]])
    res = pd.concat([data, dummies], axis=1)
    res = res.drop([feature], axis=1)
    return(res)

def label_encode(data, feature):
    le = LabelEncoder()
    le.fit([feature])
    return le.fit_transform(data)


# input data X: one-hot encoding   
fitting_features = ['HomeTeam', 'AwayTeam', 'HTR']
for ff in fitting_features:
    X = one_hot_encode(X, ff)

# ground truth y: label encoding
print(y)
#y = label_encode(y, 'FTR')

#X = X.fillna(0)    
#y_Train = np.nan_to_num(y_Train, nan=1)
print(y)

#ros = RandomOverSampler(random_state=42)
#X, y = ros.fit_resample(X, y)
'''

def random_search(X_train, y_train, n_estimators=1000, n_iter=10, cv=5):
    # creating the parameter grid with variables
    param_grid = {
        'n_estimators' : np.arange(50,200,15),
        'max_features' : np.arange(0.5, 1, 0.1),
        'max_depth' : [3, 5, 7, 9],
        'min_samples_split' : np.arange(2, 10, step=2),
        'bootstrap': [True, False]
    }

    # RandomForestClassifier selected as estimator
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(rf, param_grid, n_iter=n_iter, scoring='accuracy', n_jobs=-1, cv=5)
    rf_random.fit(X, y)
    
    best_model = rf_random.best_estimator_
    
    return best_model

model = random_search(X, y)



### testing ###

testing_path = "/content/drive/MyDrive/COMP0036CW/dataset/final_test_data.csv"
testing_data = pd.read_csv(testing_path)

# preprcessing data
X_test = testing_data.drop(['FTR'], 1)
y_test = testing_data['FTR']


print(model)
y_pred = model.predict(X_test)
score = f1_score(y_test, y_pred, average='macro')
