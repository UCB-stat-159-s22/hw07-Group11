import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


import hw7_tools.plot_tools as pt

DATA = Path.home()/ "HW" / "hw07-hw07-group11" / "data" / "application_data.csv"
print(DATA)

OUTPUT = Path.home()/ "HW" / "hw07-hw07-group11" / "output"

df = pd.read_csv(str(DATA))

def test_plot_default_in_category_group():
    pt.plot_default_in_category_group('NAME_CONTRACT_TYPE', df, 'TEST1')

    PATH = Path(str(OUTPUT) + "/TEST1.png")
    assert PATH.exists()
    
def test_plot_default_in_numerical_discrete_group():
    pt.plot_default_in_numerical_discrete_group('OBS_30_CNT_SOCIAL_CIRCLE', df, 'TEST2')
    PATH = Path(str(OUTPUT) + "/TEST2.png")
    
    assert PATH.exists()