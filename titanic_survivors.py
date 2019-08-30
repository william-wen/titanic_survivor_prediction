import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
# %matplotlib inline
# sns.set()

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print(df_train.info())
