#!/usr/bin/env python
# coding: utf-8

# # Credit Account Default Status and Characteristics

# Data source: https://www.kaggle.com/datasets/mishra5001/credit-card?datasetId=263888&select=application_data.csv (select all columns when downloading)
# 
# Detailed data and columns description and also be found using this link, under Data and Data Explorer-columns_description.csv

# Credit history matters a lot when individuals want to take out loans ... say more about this later

# In[1]:


from pathlib import Path


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


import hw7_tools.plot_tools as pt

warnings.filterwarnings("ignore")


# In[3]:


original_df = pd.read_csv('data/application_data.csv')


# In[4]:


original_df.head()


# ## Data Cleaning

# In[5]:


sorted_null = original_df.isnull().mean().sort_values(ascending=False)
sorted_null.index[(sorted_null>0.5)]


# We observe that all the above columns contain more than X%(50% in our case) of nulls values, and here we make one of our major assumptions in this EDA that these data are simply unknown and can introduce bias which can affect our analysis(as opposed to them being actually informative and showing something in our data). Future work might involve taking a deeper look into these columns, but for now, with this assumption in mind, we will drop these columns together with some additional columns like ID number, which we expect to be not useful in our analysis:

# In[6]:


# drop columns with more then 50% missing values + additional ones decided by the group
drop_col = ['SK_ID_CURR',]
df = original_df[sorted_null.index[(sorted_null<0.5)]].drop(drop_col, axis=1)


# ## EDA

# In the following parts of our analysis, we will identify some interesting findings in our data and attempt to build a logistic regression model to predict whether a customer will default or not. But for now, let us explore the different variables in our datasets.
# 
# First, we look at the overall rough distribution of several features we are initially interested in through boxplots:

# ### Data Intuition

# In[7]:


df.columns


# In[8]:


features = ['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','DAYS_ID_PUBLISH', 'DAYS_REGISTRATION']

plt.figure(figsize = (20, 15), dpi=300)
for i in enumerate(features):
    plt.subplot(3, 2, i[0]+1)
    sns.boxplot(x = i[1], data = df)
plt.show()


# While some of the distributions look reasonable, we also observe that there are obviously some strange outliers, for example in the case of days employed, there is a single point who is employed for 350000+ days, and in the AMT_TOTAL_INCOME plot, we also see a client with 100 million in income! This plot is for our initial exploratory purpose only and it perfectly illustrates one important characteristic of our dataset which is that there are errors/extreme individuals, and they might require a closer examination if we plan to use them in further analysis. 
# 
# In our case, since we plan to be using AMT_TOTAL_INCOME, we try to take into account of this issue of having extreme/unreliable values by looking at quantiles instead to minimize the impact of such outliers.

# ### Target

# Next, we then proceed to looking at the general picture of what is the proportion of people using credit cards who are able to pay on time vs those who failed to pay on time("TARGET" column).

# In[9]:


sns.barplot(x="TARGET", y="TARGET", data=df, estimator=lambda x: len(x) / len(df) * 100)
plt.xlabel("Default")
plt.ylabel("% of customers")
plt.title("Distribution of TARGET Variable(0 for non-default, 1 for default)")
plt.show()
print("proportion of people who paid on time:", 1-df["TARGET"].mean())
print("proportion of people who failed to pay on time:", df["TARGET"].mean())


# Notice that there the original data has a high class imbalance, so we might want to bear this fact in mind when performing logistic regression and considering the metrics to look at when evaluating classifiers later. 

# ## Ratio of Default vs Non-Default for different groups
# 
# 
# We will now examine the ratio of defaulters vs non-defaulters between some other categorical variables' groups 

# ### Gender

# In[10]:


# omit the several XNA gender entries since we want to avoid assuming one's gender
gender_df = df[df['CODE_GENDER']!= 'XNA']
pt.plot_default_in_category_group('CODE_GENDER', gender_df, 'Gender')


# **Observations:** 
# * Males seem to have a slightly higher proportion of defaults compared to female

# #### Gender and Income

# To further investigate the relationship between default risk and gender, we will manually divide income into 5 groups by percentile ranking and see how gender and default rate varies between these different income groups:

# In[14]:


# binning AMT_INCOME_TOTAL column based on quantiles
order = ['VERY LOW','LOW','MEDIUM','HIGH','VERY HIGH']
gender_df['AMT_INCOME_Quantile'] = pd.qcut(gender_df.AMT_INCOME_TOTAL, q=[0,0.2,0.4,0.6,0.8,1], labels=order)
df_one = gender_df[gender_df['TARGET']==1]

f, ax = plt.subplots(1,figsize=(12,6))
sns.histplot(ax = ax,x= df_one['AMT_INCOME_Quantile'],hue = df_one['CODE_GENDER'],discrete = True)
ax.set_xlabel("Income Quantile")
ax.set_title("Count of Defaults of F/M in Different Income Groups");


# This might be surprising, but we observe that females have an overall higher count of defaults, and in the occasions where males have a higher default count in the higher income groups, they only "win" by a slight amount. This is opposite to what we observed in the previous section, where males were observed to have a slightly higher proportion of defaults compared to female. The reason is that, if you remembered, there are almost **double the amount of female entries vs male entries** in our original dataset!

# Given the class imbalance in gender, we created a plotting function that generates the proportion of defaults of each gender, accounted for the different total counts in gender, and now this new proportion plot give a much more accurate depiction of our actual data.

# In[13]:


pt.balanced_plot(gender_df)


# **Observations:** 
# * We now see that, indeed, males seem to have a overall higher rate of default, and this is especially the case in higher income groups, as the gap keeps increasing with income and doubles in the two highest income groups. This is a very interesting observation and further studies might be conducted on this in the future to explore what exactly is contributing to this observation. 

# ### Loan Type

# In[33]:


pt.plot_default_in_category_group('NAME_CONTRACT_TYPE', df, 'Loan')


# **Observations:** 
# * People who take out cash loans have a slightly higher rate of defaulting compared to revolving loans

# ### Income Type

# In[15]:


pt.plot_default_in_category_group('NAME_INCOME_TYPE', df, 'Income Type')


# **Observations:** 
# * Unemployed and Maternity leave groups seem to have a much higher proportion of default, whereas state servents and pensioners tend to have a lower proportion of defaults among them.

# ### Education Type

# In[16]:


pt.plot_default_in_category_group('NAME_EDUCATION_TYPE', df, 'Education')


# **Observations:** 
# * The academic degree and higher education group have a lower proportion of defaults compared to the other groups that are slightly lower in educational degree status

# ### House Ownership

# In[17]:


pt.plot_default_in_category_group('FLAG_OWN_REALTY', df, 'House Ownership')


# **Observations:** 
# * Whether someone owns a house or not does not change significantly the rates of defaulting on a loan

# ### House Type

# In[18]:


pt.plot_default_in_category_group('NAME_HOUSING_TYPE', df, 'House Type')


# **Observations:** 
# * Despite significant class imbalance where the vast majority of people have a home/apartment, the highest rates of defaulting in the house type category are those who live in a rented apartment. Higher default rates are also seen for those that live with their parents. 

# ### Number of Surrounding 30DPD Defaults

# In[19]:


pt.plot_default_in_numerical_discrete_group('OBS_30_CNT_SOCIAL_CIRCLE', df, 'Number of Surrounding 30DPD Defaults')


# **Observations:** 
# * With the increase in the number of surrounding 30day default cases, we also see a general increase in default proportion. 

# ### Number of Family Members

# In[39]:


pt.plot_default_in_numerical_discrete_group('CNT_FAM_MEMBERS', df, 'Number of Family Members')


# **Observations:** 
# * With the increase in the number of family numbers, we also see a slight upward trend in default proportion. 

# #### Data Takeaways

# The plots above give a clear picture about the type of people within the data. The majority of people are working, female, own a home, are living in a home/apartment, and have 0 surrounding 30DPD defaults. It can also be seen that the max education for the majority of people in the data is a secondary education, which in the USA, refers to completing high school. It is interesting to see just how many people in the data own an apartment or house as it is significantly higher than for example the number of people renting. With that said, the data population really is not representative of the overall population as a whole. 
# 
# *Note that we did remove the extreme outliers on the right end for the above two discrete data plotting in order to achieve better visualization clarify, and future work might include more detailed analysis on the outliers.

# ## Logistic Regression Model

# The modeling section focuses on building a logistic regression model to try and predict whether someone will default on a loan or not given particular parameters from the data. As seens in the previous section, there is significant class imbalance where the vast majority of people in the data do not default on their loan. Thus metrics such as recall and precision are more important tools to determine the validity of our model compared to accuracy.

# In[40]:


import sklearn.linear_model as lm

SelectedCol =['TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                   'FLAG_OWN_CAR','NAME_FAMILY_STATUS','OBS_30_CNT_SOCIAL_CIRCLE','CNT_FAM_MEMBERS',
                  'FLAG_OWN_REALTY']
Features = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                   'FLAG_OWN_CAR','NAME_FAMILY_STATUS','OBS_30_CNT_SOCIAL_CIRCLE','CNT_FAM_MEMBERS',
                  'FLAG_OWN_REALTY']


# In[41]:


# preprocessing gender and flag_own_car 
categorical_cols = ['FLAG_OWN_CAR', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE', 'FLAG_OWN_REALTY','NAME_FAMILY_STATUS']
processed = pd.get_dummies(df[SelectedCol], columns = categorical_cols)


# In[42]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(processed, test_size=0.2,random_state = 1)
train_1 = train[train['TARGET']==1].sample(10000, random_state = 1)
train_0 = train[train['TARGET']==0].sample(10000, random_state = 1)
balanced_train = pd.concat([train_1,train_0])
balanced_train_x = balanced_train[balanced_train.columns[~balanced_train.columns.isin(['TARGET'])]]
balanced_train_y = balanced_train['TARGET']
test_x = test[test.columns[~test.columns.isin(['TARGET'])]]
test_y = test['TARGET']


# In[43]:


model = lm.LogisticRegression(penalty = 'none', fit_intercept = True, solver = 'lbfgs', random_state = 1)


# In[44]:


model.fit(balanced_train_x, balanced_train_y)


# In[45]:


################
model.predict_proba(balanced_train_x)


# In[46]:


# displaying which coefficients are with which columns

coefficients = pd.DataFrame({'feature': balanced_train_x.columns, 'coeff': model.coef_[0]}, columns=['feature', 'coeff'])
coefficients.sort_values(by='coeff').head()


# In[47]:


# accuracy

y_pred = model.predict(balanced_train_x)

np.mean(y_pred == balanced_train_y)


# In[48]:


from sklearn.metrics import confusion_matrix


# In[49]:


cm = confusion_matrix(balanced_train_y, y_pred)
sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', annot_kws = {'size': 16})
plt.xlabel('Predicted')
plt.ylabel('Actual');


# In[50]:


tp = np.sum((y_pred == 1) & (balanced_train_y == 1))
tn = np.sum((y_pred == 0) & (balanced_train_y == 0))
fp = np.sum((y_pred == 1) & (balanced_train_y == 0))
fn = np.sum((y_pred == 0) & (balanced_train_y == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f'precision = {precision}')
print(f'recall = {recall}')


# ### Test Results

# In[51]:


y_pred = model.predict(test_x)
np.mean(y_pred == test_y)


# In[52]:


cm_test = confusion_matrix(test_y, y_pred)
sns.heatmap(cm_test, annot=True, fmt = 'd', cmap = 'Blues', annot_kws = {'size': 16})
plt.xlabel('Predicted')
plt.ylabel('Actual');


# In[ ]:


tp = np.sum((y_pred == 1) & (test_y == 1))
tn = np.sum((y_pred == 0) & (test_y == 0))
fp = np.sum((y_pred == 1) & (test_y == 0))
fn = np.sum((y_pred == 0) & (test_y == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f'precision = {precision}')
print(f'recall = {recall}')


# Maximizing precision will minimize the number false positives, whereas maximizing the recall will minimize the number of false negatives.

# ## *backup/extra code

# In[ ]:


# # Finding the unique values for columns with categorical variables
# unique_df = categorical_df.nunique().reset_index()
# unique_df.columns = ['Column','Number of Unique Values']

# # extract unique classes in each column
# unique_df['Unique Values'] = unique_df['Column'].apply(lambda x: df[x].unique())
# unique_df.sort_values(by='Number of Unique Values')


# In[ ]:


#correlation matrix 
#corr = numerical_df.corr()
#corr.style.background_gradient(cmap='coolwarm')
#corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# plt.savefig('output/correlation_matrix.png') # not sure how to save this figure


# In[ ]:


# how target variable varies depending on contract type - does this make sense? 

# ax = sns.barplot(x="NAME_CONTRACT_TYPE", y="TARGET", data=df)
# plt.title('Target Variable Count Based on Contract Type')
# plt.savefig('output/target_contract_type_counts.png')


# In[ ]:


# Gender balance

# ax = sns.barplot(x=['M', 'F'], y=[sum(df['CODE_GENDER']=='M'), sum(df['CODE_GENDER']=='F')])
# plt.title('Gender Counts')
# plt.savefig('output/gender_counts.png')


# In[ ]:


# df['YEARS_BIRTH'] = -df['DAYS_BIRTH']/365.25
# df['YEARS_BIRTH'] = np.floor(df['YEARS_BIRTH'].copy())

# # plot the distribution of ages 

# sns.histplot(data=df, x="YEARS_BIRTH")
# plt.title('Distribution of Age')
# plt.xlabel('Age in Years')
# plt.ylabel('Freq')
# plt.savefig('output/age_counts.png')


# In[ ]:


#for column in numerical_df.columns:
#    plt.figure(figsize=(20,5))
#    plt.subplot(121)
#    sns.distplot(df[column])
#    plt.title(column)
#    # save figure


# In[ ]:


#for column in categorical_df.columns:
#    plt.figure(figsize=(30,6))
#    plt.subplot(121)
#    sns.countplot(x=df[column],hue=df['TARGET'],data=df)
#    plt.title(column)    
#    plt.xticks(rotation=90)
#    # save figures


# In[ ]:





# In[ ]:





# In[ ]:




