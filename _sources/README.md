## Welcome to HW 7

## Credit Account Default Status and Characteristics

**[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw07-Group11.git/HEAD?labpath=main.ipynb)**


**Note:** This repository is public. The [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mishra5001/credit-card?datasetId=263888&select=columns_description.csv) is from Kaggle. The EDA and Logistic Regression Model were developed by Joe, Isaac, and Uma as a homework assignment for the [Spring 2022 installment of UC Berkeley's Stat 159/259 course, _Reproducible and Collaborative Data Science](https://ucb-stat-159-s22.github.io).

In our studies, we will take a look at credit risk and how default status varies within different groups of people with different characteristics. We will not only try to find the relationship of some of our selected variables like how does default rate varies among gender or income classes, but will also build and train a simple logistic regression model that can help us predict whether a client is likely to default or not. The entire analysis is contained in **main.ipynb**

### Some helpful commands:

Install the packages:
``pip install .`` *or* ``make hw7_tools``

Test the packages:
``pytest hw7_tools``

Generate the minimum required environment:
``make env``

Execute and generate outputs: 
``make all``

Clean outputs:
``make clean``

### Some helpful tips:

After ``make clean``, rerun details.ipynb or ``make all`` to regenerate the data and outputs needed for main.ipynb

Detailed data computations of processing/plotting/modeling/serialization can be found in *details.ipynb*, if you are curious and want to play with our results(e.g. by adjusting data/parameters), please also look into *details.ipynb* and do not work in *main.ipynb* since all outputs are loaded into *main.ipynb* instead of being computed there.
