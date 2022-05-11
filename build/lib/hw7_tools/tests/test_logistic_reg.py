import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

import hw7_tools.logistic_reg as logi
DATA = os.getcwd() + "/data/application_data.csv"
original_df = pd.read_csv(str(DATA))
sorted_null = original_df.isnull().mean().sort_values(ascending=False)
sorted_null.index[(sorted_null>0.5)]
drop_col = ['SK_ID_CURR',]
df =original_df[sorted_null.index[(sorted_null<0.5)]].drop(drop_col, axis=1)
SelectedCol =['TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                   'FLAG_OWN_CAR','NAME_FAMILY_STATUS','OBS_30_CNT_SOCIAL_CIRCLE','CNT_FAM_MEMBERS',
                  'FLAG_OWN_REALTY']
Features = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                   'FLAG_OWN_CAR','NAME_FAMILY_STATUS','OBS_30_CNT_SOCIAL_CIRCLE','CNT_FAM_MEMBERS',
                  'FLAG_OWN_REALTY']

categorical_cols = ['FLAG_OWN_CAR', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE', 'FLAG_OWN_REALTY','NAME_FAMILY_STATUS']
processed = pd.get_dummies(df[SelectedCol], columns = categorical_cols)

def test_logi_output_type():
    coef = logi.logistic_reg(processed)
    assert(type(coef['feature'][0])==str)
    assert(type(coef['coeff'][0])==np.float64)