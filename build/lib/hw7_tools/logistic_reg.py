from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def logistic_reg(processed):
    #solving imbalance
    threshold = 0.55 
    train, test = train_test_split(processed, test_size=0.2,random_state = 1)
    train_1 = train[train['TARGET']==1].sample(10000, random_state = 1)
    train_0 = train[train['TARGET']==0].sample(10000, random_state = 1)
    balanced_train = pd.concat([train_1,train_0])
    balanced_train_x = balanced_train[balanced_train.columns[~balanced_train.columns.isin(['TARGET'])]]
    balanced_train_y = balanced_train['TARGET']
    test_x = test[test.columns[~test.columns.isin(['TARGET'])]]
    test_y = test['TARGET']
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
    #model
    model = lm.LogisticRegression(penalty = 'none', fit_intercept = True, solver = 'lbfgs', random_state = 1)
    model.fit(balanced_train_x, balanced_train_y)
    prob = model.predict_proba(balanced_train_x)
    y_pred = []
    for x in prob:
        if (x[0]<0.55):
            y_pred.append(1)
        else:
            y_pred.append(0)
    #display coefficients
    coefficients = pd.DataFrame({'feature': balanced_train_x.columns, 'coeff': model.coef_[0]}, columns=['feature', 'coeff'])
    coefficients.sort_values(by='coeff').head() 
    
    cm = confusion_matrix(balanced_train_y, y_pred)
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', annot_kws = {'size': 16},ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Predictions on Train Set');
    
    #summary stats
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (balanced_train_y == 1))
    tn = np.sum((y_pred == 0) & (balanced_train_y == 0))
    fp = np.sum((y_pred == 1) & (balanced_train_y == 0))
    fn = np.sum((y_pred == 0) & (balanced_train_y == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    
    prob = model.predict_proba(test_x)
    y_pred = []
    for x in prob:
        if (x[0]<threshold):
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    # results on testing set
    cm2 = confusion_matrix(test_y, y_pred)
    sns.heatmap(cm2, annot=True, fmt = 'd', cmap = 'Blues', annot_kws = {'size': 16},ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Predictions on Test Set');
    tp = np.sum((y_pred == 1) & (test_y == 1))
    tn = np.sum((y_pred == 0) & (test_y == 0))
    fp = np.sum((y_pred == 1) & (test_y == 0))
    fn = np.sum((y_pred == 0) & (test_y == 1))
    test_precision = tp / (tp + fp)
    test_recall = tp / (tp + fn)
    coefficients = pd.DataFrame({'feature': balanced_train_x.columns, 'coeff': model.coef_[0]}, columns=['feature', 'coeff'])
    print(f'train_precision = {precision} vs test_precision = {test_precision}')
    print(f'train_recall = {recall} vs test_recall = {test_recall}')
    coefficients = pd.DataFrame({'feature': balanced_train_x.columns, 'coeff': model.coef_[0]}, columns=['feature', 'coeff'])
    return coefficients

