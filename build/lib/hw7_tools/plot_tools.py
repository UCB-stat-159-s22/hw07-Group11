import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

OUTPUT = os.getcwd() + "/output"

def plot_default_in_category_group(group_name, df, x_label):
    """Add docstrings"""
    
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
    ax.set_title("Default Proportions(Imbalance Accounted) of F/M in Different Income Groups");
