import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix
import os

OUTPUT = os.getcwd() + "/output"

def plot_default_in_category_group(group_name, df, x_label):
    """This function helps create two distribution plots side by side. Generates a pink plot(left) showing the distribution of the variable and a blue plot(right) showing the proportion of defaults in different groups of the chosen variable.

    Parameters
    ----------
    group_name : str
        This specifies which column's distribution will be plotted.
    df : DataFrame
        The input dataframe where we use as our source dataset for the columns to plot.
    x_label : str
        The name we would want to use when we specify the title, usually a more readable text of group_name.
    """
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
    title = 'Default Distribution Among Different ' + x_label +' Group'
    ordering =  list(df[group_name].unique())
    
    sns.histplot(ax=ax1, x=group_name, data=df, stat='density', color='pink')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('percent')
    ax1.set_title("Overall Distribution for each " + x_label + ' Group')

    grouped = (df.groupby([group_name,'TARGET']).size() / df.groupby([group_name]).size()).reset_index().rename({0:'percent'}, axis=1)
    # renaming TARGET into default for clarity
    grouped = grouped[grouped['TARGET']==1].rename(columns = {'TARGET':"Default"})
    sns.barplot(ax = ax2, x=group_name, hue="Default", y='percent', data=grouped,order=ordering)
    ax2.set_xlabel(x_label)
    ax2.set_title("Default Distribution for each " + x_label + ' Group')
    
    if (len(ordering) >= 5):
        ax1.set_xticklabels(ordering, rotation = 75)
        ax2.set_xticklabels(ordering, rotation = 75)
    f.suptitle(title)
    f.tight_layout()
    plt.savefig(str(OUTPUT) + "/" + x_label + ".png")
    
    
def plot_default_in_numerical_discrete_group(group_name, df, x_label):
    """Add docstrings"""
    filled = df[[group_name]].fillna(value=0)
    df[group_name] = filled
    # removing extreme outliers and calculating an appropriate binwidth
    sorted_df = list(df.sort_values(by=[group_name],ascending=False).index)[1000:]
    new_df = df.query('index in @sorted_df')
    new_df = new_df.rename(columns = {'TARGET':"Default"})
    
    
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
    sns.histplot(data = new_df,x=group_name,binwidth=1, ax=ax1, color = 'pink', stat = 'density')
    
    ax1.set_xlabel(x_label)
    ax1.set_title('Distribution of '+x_label)
    
    prop = new_df[[group_name,'Default']].groupby(group_name).mean().reset_index()
    sns.lineplot(data = prop,x = group_name,y = 'Default', ax=ax2, color = 'blue')
    ax2.set_ylabel('Default Proportion')
    ax2.set_xlabel(x_label)
    ax2.set_title('Default Proportion vs '+ x_label)
    f.tight_layout()
    plt.savefig(str(OUTPUT) + "/" + x_label + ".png")

def balanced_plot(gender_df):
    order = ['VERY LOW','LOW','MEDIUM','HIGH','VERY HIGH']
    df_one = gender_df[gender_df['TARGET']==1]
    
    total_f = len(gender_df[gender_df['CODE_GENDER']=='F'])
    total_m = len(gender_df['CODE_GENDER'])-total_f
    
    grouped_df = df_one.groupby(['AMT_INCOME_Quantile','CODE_GENDER']).count()[['REG_CITY_NOT_LIVE_CITY']].rename(columns={
        "REG_CITY_NOT_LIVE_CITY": "count"}).reset_index()
    
    accounted_for_imbalance = [None] * 10
    for i in range(10):
        ct = grouped_df['count']
        if((i+1)%2 != 0):
            accounted_for_imbalance[i] = ct[i]/total_f
        else:
            accounted_for_imbalance[i] = ct[i]/total_m                                      
    grouped_df['balanced_count']=accounted_for_imbalance
    
    f, ax = plt.subplots(1,figsize=(12,6))
    sns.barplot(ax = ax,data = grouped_df,x = 'AMT_INCOME_Quantile',y='balanced_count',hue= 'CODE_GENDER')
    ax.set_xlabel("Income Quantile")
    ax.set_title("Default Proportions(Imbalance Accounted) of F/M in Different Income Groups")
    plt.savefig("output/balanced_plot.png");
    return accounted_for_imbalance

    
def logistic_reg(processed, threshold=0.5):
    """This is a function used for doing logistic regression fitting, predicting, and results analyzing via plots and summary stats. Plots include confusion matrix for the training and testing sets, with the respective recall and precision values printed above the plots

    Parameters
    ----------
    processed : DataFrame
        The processed dataframe ready for logistic regression(cleaned, dummy ready datasets)
    threshold: float
        The manually set threshold used for our decisions on how we want to balance precision and recall. Default is 0.5.

    Returns
    -------
    DataFrame
        A dataframe that records the coefficients generated by sklearns' logistic regression function, records the coefficient for each respective feature.

    Raises
    ------
    ValueError
        If threshold is not in the range of (0,1).
    """
    #solving imbalance
    if ((threshold>=1) or (threshold<=0)) :
        raise ValueError('threshold must be between 1 and 0')
    th = threshold
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
        if (x[0]<th):
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
        if (x[0]<th):
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
    plt.savefig("output/reg_matrix.png")
    return coefficients

def feature_plot(features,df):
    plt.figure(figsize = (20, 15), dpi=300)
    for i in enumerate(features):
        plt.subplot(3, 2, i[0]+1)
        sns.boxplot(x = i[1], data = df)
    plt.savefig("output/feature_plot.png")
    plt.show()

def target_plot(df):
    sns.barplot(x="TARGET", y="TARGET", data=df, estimator=lambda x: len(x) / len(df) * 100)
    plt.xlabel("Default")
    plt.ylabel("% of customers")
    plt.title("Distribution of TARGET Variable(0 for non-default, 1 for default)")
    plt.show()
    plt.savefig("output/target_plot.png")
    print("proportion of people who paid on time:", 1-df["TARGET"].mean())
    print("proportion of people who failed to pay on time:", df["TARGET"].mean())
    
def gender_income_plot(gender_df):
    # binning AMT_INCOME_TOTAL column based on quantiles
    order = ['VERY LOW','LOW','MEDIUM','HIGH','VERY HIGH']
    gender_df['AMT_INCOME_Quantile'] = pd.qcut(gender_df.AMT_INCOME_TOTAL, q=[0,0.2,0.4,0.6,0.8,1], labels=order)
    df_one = gender_df[gender_df['TARGET']==1]
    
    f, ax = plt.subplots(1,figsize=(12,6))
    sns.histplot(ax = ax,x= df_one['AMT_INCOME_Quantile'],hue = df_one['CODE_GENDER'],discrete = True)
    ax.set_xlabel("Income Quantile")
    ax.set_title("Count of Defaults of F/M in Different Income Groups");
    plt.savefig("output/gender_income_plot.png")
    plt.show()