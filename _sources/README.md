## Credit Account Default Status and Characteristics

**[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw07-Group11/HEAD?labpath=main.ipynb)**

**Note:** This repository is public. The [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mishra5001/credit-card?datasetId=263888&select=columns_description.csv) is from Kaggle. The EDA and Logistic Regression model were developed by Joe, Isaac, and Uma as a homework assignment for the [Spring 2022 installment of UC Berkeley's Stat 159/259 course, _Reproducible and Collaborative Data Science](https://ucb-stat-159-s22.github.io).

## Makefile
The following are the available make commands:
- `env`: Creates the environment and configures it by activating it and installing ipykernel into it
- `hw7_tools`: Installs the hw07 tools package
- `clean`: Remove all figures from the /output directory

## File Structure
`details.ipynb` and `main.ipynb` contain the same outputs; however, `details.ipynb` computes all the output, while `main.ipynb` loads the images directly. Data can be found in `data/application_data.csv`, and `hw7_tools` contains the functions used in the notebook as well as tests for those functions.