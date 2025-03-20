#import libaries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

#load data
loan_original = pd.read_csv("data/loan_approval_dataset.csv")

#remove unwanted spaces
loan_original.columns = loan_original.columns.str.replace(' ', '')

#unwanted column
loan = loan_original.drop(['loan_id'], axis=1)

loan_dummies = pd.get_dummies(loan)
loan_dummies.rename(columns = {'education_ Graduate':'education', 'self_employed_ Yes':'self_employed', 'loan_status_ Approved':'loan_status' }, inplace = True)
loan_dummies = loan_dummies.drop(['education_ Not Graduate', 'self_employed_ No', 'loan_status_ Rejected'], axis=1)

#data splitting
y = loan_dummies['loan_status']
X = loan_dummies.drop(['loan_status'], axis =1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y,  test_size = 0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

stand_scaler = StandardScaler()
stand_scaler.fit(X_train)
X_train_stand = stand_scaler.transform(X_train)
X_val_stand = stand_scaler.transform(X_val)
X_test_stand = stand_scaler.transform(X_test)

#(a)Logistic Regression
clf = LogisticRegression().fit(X_train_stand,y_train)
y_lr = clf.predict(X_val_stand) 

#Calculate LR Model Scores
#print('Accuracy:', '%.3f' % accuracy_score(y_val, y_lr)) #91.8%
#print('Precision:', '%.3f' % precision_score(y_val, y_lr)) #93.5%
#print('Recall:', '%.3f' % recall_score(y_val, y_lr)) #92.7%
#print('F1 Score:', '%.3f' % f1_score(y_val, y_lr)) #93.1%

#(b)Random Forest
rf_opt = RandomForestClassifier(n_estimators = 150, max_depth = None, 
                                min_samples_leaf = 1, min_samples_split = 5,random_state = 0)
rf_opt.fit(X_train, y_train)
y_rf = rf_opt.predict(X_val)

#Calculate RF Model Scores
#print('Accuracy:', '%.3f' % accuracy_score(y_val, y_rf)) #97.2%
#print('Precision:', '%.3f' % precision_score(y_val, y_rf)) #97.6%
#print('Recall:', '%.3f' % recall_score(y_val, y_rf)) #97.6%
#print('F1 Score:', '%.3f' % f1_score(y_val, y_rf)) #97.6%

y_test_rf = rf_opt.predict(X_test)

#print('Accuracy:', '%.3f' % accuracy_score(y_test, y_test_rf)) #97.3%
#print('Precision:', '%.3f' % precision_score(y_test, y_test_rf)) #97.3%
#print('Recall:', '%.3f' % recall_score(y_test, y_test_rf)) #97.9
#print('F1 Score:', '%.3f' % f1_score(y_test, y_test_rf)) #97.9

#Since Random Forest model performed better performance than the Logistic Regression Model, we are going to use this one

import pickle

# Save the column names
with open("columns.pkl", "wb") as file:
    pickle.dump(X_train.columns, file)

# Save the RF model
with open("RF_model.pkl", "wb") as file:
    pickle.dump(rf_opt, file)