import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

import hw7_tools.plot_tools as pt

DATA = os.getcwd() + "/data/serialized.fth"

OUTPUT = os.getcwd() + "/output"

df = pd.read_feather(DATA)

gender_df = df[df['CODE_GENDER']!= 'XNA']
pt.plot_default_in_category_group('CODE_GENDER', gender_df, 'Gender')
order = ['VERY LOW','LOW','MEDIUM','HIGH','VERY HIGH']
gender_df['AMT_INCOME_Quantile'] = pd.qcut(gender_df.AMT_INCOME_TOTAL, q=[0,0.2,0.4,0.6,0.8,1], labels=order)
df_one = gender_df[gender_df['TARGET']==1]

def test_plot_default_in_category_group():
    pt.plot_default_in_category_group('NAME_CONTRACT_TYPE', df, 'TEST1')

    PATH = Path(str(OUTPUT) + "/TEST1.png")
    assert PATH.exists()
    
def test_plot_default_in_numerical_discrete_group():
    pt.plot_default_in_numerical_discrete_group('OBS_30_CNT_SOCIAL_CIRCLE', df, 'TEST2')
    PATH = Path(str(OUTPUT) + "/TEST2.png")
    assert PATH.exists()

def test_balanced_plot_output():
    balanced = pt.balanced_plot(gender_df)
    # testing that all new proportions are actually less than 1(aka indeed ratios)
    list_tf = np.array(balanced)<1
    assert(sum(list_tf)==10)
    