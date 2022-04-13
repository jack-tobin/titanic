#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:33:37 2022

@author: jtobin
"""

#%% Setup environment
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# fun stuff
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier

# change directory
os.chdir(os.path.expanduser('~/Documents/projects/titanic'))

#%% Read in data and prelim cleaning

def clean_data(df, training=True):
    # set index
    df.set_index('PassengerId', inplace=True)

    # unpack name
    data['Surname'] = data['Name'].str.split(',', 1, expand=True)[0]
    data['FullFirstName'] = data['Name'].str.split(',', 1, expand=True)[1].str.lstrip()
    data['Title'] = data['FullFirstName'].str.split('.', 1, expand=True)[0]
    data['FirstName'] = data['FullFirstName'].str.split('.', 1, expand=True)[1].str.lstrip()
    data.drop(['FullFirstName'], axis=1, inplace=True)

    # fill missing values for embarked with final destination, encode
    data['Embarked'].fillna('C', inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # dummy for male/female
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # impute missing values for age based on other features
    imp = IterativeImputer()
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    data[cols] = imp.fit_transform(data[cols])

    # drop cabin since too many NaNs
    data.drop(['Cabin'], axis=1, inplace=True)
    
    # Fare: log
    data['Fare'].replace(0, 1, inplace=True)
    data['Fare'] = np.log(data['Fare'])

    # decile for age
    data['Age_q'] = pd.qcut(data['Age'], 10, labels=False)
    
    # decile for fare
    data['Fare_q'] = pd.qcut(data['Fare'], 10, labels=False)
    
    # select X and y
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    if training:
        y = data['Survived'].to_numpy().reshape(-1,1)
        return data, X, y
    else:
        return data, X


# load in training data
data = pd.read_csv('data/train.csv')

# clean up data
data, X, y = clean_data(data, training=True)

#%% Visuals

# plot scatter matrix
# pd.plotting.scatter_matrix(data)

# investigate survival rates by class, sex
# From: https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
# fig, ax = plt.subplots(1)
# sns.barplot(data=data, x='Pclass', y='Survived', hue='Sex', ax=ax)
    
# looks like women were many times more likely to survive than men regardless
# of class; more women died in third class than first and second class

# investigate survival rates by age distribution
# grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)

# first class deaths were evenly distributed; almost no children in 2nd class
# died; deaths in third class were mostly younger

#%% Begin modelling

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# import binary classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# arrange into list for iterative modelling
models = {'knn': KNeighborsClassifier(),
          'tree': DecisionTreeClassifier(),
          'forest': RandomForestClassifier(),
          'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'lda': LinearDiscriminantAnalysis()}

# run through models
for k, v in models.items():
    # fit model
    v.fit(X_train, y_train.ravel())
    
    # predict y's
    pred = v.predict(X_test)
    
    # accuracy score
    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    
    # print results
    print(k + '; Accuracy: ' + '{:.1%}'.format(acc) + '; Recall: ' + '{:.1%}'.format(rec))
    
#%% Focus on Logit model
    
# logistic regression has highest fit, begin fine tuning of parameters
logit = LogisticRegression()
logit.fit(X_train, y_train.ravel())
y_pred = logit.predict(X_test)
print('Sensitivity: {:.1%}'.format(accuracy_score(y_test,y_pred)))
print('Specificity: {:.1%}'.format(recall_score(y_test,y_pred)))

# read in testing data and print final answers
test = pd.read_csv('data/test.csv')

# clean



