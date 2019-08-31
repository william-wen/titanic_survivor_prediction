import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer

# Figures inline and set visualization style
# %matplotlib inline
# sns.set()

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
combine = [train_df, test_df]

# TELLS US DATATYPES
# print(train_df.info()) 

# NUMERICAL FEATURES
# print(train_df.describe()) 

# CATEGORICAL FEATURES
# MAKE HYPOTHESIS ON WHAT TO INCLUDE, CORRECT, AND CHANGE
# print(train_df.describe(include=['O']))

# ANALYZE CORRELATION BETWEEN CATEGORICAL DATA AND SURVIVED
# MAKE DECISIONS ON WHAT TO INCLUDE, CORRECT, AND CHANGE
# print(train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False). mean().sort_values(by='Survived', ascending=False))
# print(train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# CORRELATING NUMERICAL FEATURES (AGE AND SURVIVAL)
# g = sns.FacetGrid(train_df, col="Survived")
# g.map(plt.hist, 'Age', bins=20)

# CORRELATING NUMERICAL AND ORDINAL FEATURES (AGE AND PCLASS)
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()

# CORRELATING CATEGORICAL FEATURES 
# grid = sns.FacetGrid(train_df, row='Embarked')
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()

# CORRELATING EMBARATION, SEX, FARE, AND SURVIVAL
# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()

# ------------------  DATA PREPROCESSING ---------------------------------

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# GET TITLES
for dataset in combine: 
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# HOW MANY PEOPLE WITH EACH TITLE    
# print(pd.crosstab(train_df['Title'], train_df['Sex']))

# REPLACE WITH MORE COMMON TITLES
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# PERCENT OF PEOPLE WHO SURVIVED WITH EACH TITLE
# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# MAP ORDINAL VALUES TO THE TITLES
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# DROP NAME AND PASSENGERID
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# CONVERT NOMINAL DATA TO NUMERICAL VALUES
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# CHECK AGE CORRELATIONS WITH PCLASS AND SEX
# grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()

# FILL IN MISSING VALUES FOR AGE 
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5) * 0.5
    
    for i in range(0,2):
        for j in range (0,3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),
                        'Age'] = guess_ages[i,j]
    
    dataset['Age'] = dataset['Age'].astype(int)

# CATEGORIZE AGES INTO 5 EQUAL RANGES
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

# COMBINING PARCH AND SIBSP TO MAKE FAMILY SIZE

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# NO NUMERICAL CORRELATION
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
# print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
    
# REPLACE MISSING VALUES IN EMBARKED WITH MOST FREQUENTLY APPEARING CHARACTER
freq_port = train_df.Embarked.dropna().mode()[0] # returns a Series

# FILLS NAN's WITH A VALUE
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

# print(train_df.groupby('Embarked').count().Survived)
    
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# WHY DO WE CREATE BANDS

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# WHY DO WE NEED TO CONVERT THE NUMERICAL DATA INTO ORDINAL DATA??? 

print(test_df.head(10))

# --------------------------MODEL----------------------------------------------