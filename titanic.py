#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:33:37 2022

@author: jtobin
"""

# %% Set up environment

# basic stuff
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# fun stuff
# for imputing missing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# for model selection and testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# change directory
os.chdir(os.path.expanduser('~/Documents/projects/titanic'))

# %% Read in data and perform data cleaning


def clean_data(df, training=True):
    """
    Cleans raw passenger data.

    Parameters
    ----------
    df : DataFrame
        the raw dataframe of passenger data.
    training : Boolean, optional
        Whether the raw data is from the Kaggle training or testing dataset. 
        The default is True.

    Returns
    -------
    df : DataFrame
        DESCRIPTION.
    X : Numpy Array
        Matrix of final columns selected for machine learning.
    y : Numpy Array
        Column vector of y's

    """
    
    # set PassengerID as index
    df.set_index('PassengerId', inplace=True)

    # Unpack name for potential further future feature engineering.
    df['Surname'] = df['Name'].str.split(',', 1, expand=True)[0]
    df['FullFirstName'] = df['Name'].str.split(',', 1, expand=True)[1].str.lstrip()
    df['Title'] = df['FullFirstName'].str.split('.', 1, expand=True)[0]
    df['FirstName'] = df['FullFirstName'].str.split('.', 1, expand=True)[1].str.lstrip()
    df.drop(['FullFirstName'], axis=1, inplace=True)

    # fill missing values for embarked with final destination, encode
    df['Embarked'].fillna('C', inplace=True)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # dummy for male/female
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # impute missing values for age based on other features
    imp = IterativeImputer()
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df[cols] = imp.fit_transform(df[cols])

    # drop cabin since too many NaNs
    df.drop(['Cabin'], axis=1, inplace=True)

    # Fare: log
    df['Fare'].replace(0, 1, inplace=True)
    df['Fare'] = np.log(df['Fare'])

    # select X and y
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    if training:
        y = df['Survived'].to_numpy().reshape(-1, 1)
        return df, X, y
    else:
        return df, X


# Data downloaded via Kaggle API:
# cd data
# kaggle competitions download -c titanic

# load in training data
data = pd.read_csv('data/train.csv')

# clean up data
data, X, y = clean_data(data, training=True)

# %% Preliminary data analysis to get an idea of general takeaways.

# scatter matrix to show high level
pd.plotting.scatter_matrix(data)
plt.gcf().savefig('scatterMatrix.png')
plt.close()

# investigate survival rates by class, sex
# inspired by: https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
fig, ax = plt.subplots(1)
sns.barplot(data=data, x='Pclass', y='Survived', hue='Sex', ax=ax)
plt.gcf().savefig('barplot.png')
plt.close()

# Looks like women were many times more likely to survive than men regardless
# of class; more women died in third class than first and second class.

# investigate survival rates by age distribution
grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
plt.gcf().savefig('surviveGrid.png')
plt.close()

# First class deaths were evenly distributed as first class was more likely
# to survive in general; almost no children in 2nd class died; deaths in 
# third class were mostly younger people, likely more younger people in 
# third class to begin with.

# %% Begin modelling

# split data for training and testing purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Arrange models into dict for iterative modelling.
models = {'knn': KNeighborsClassifier(),
          'tree': DecisionTreeClassifier(),
          'forest': RandomForestClassifier(),
          'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'lda': LinearDiscriminantAnalysis()}

# run through models: fit, predict y's, print accuracy and recall scores
for k, v in models.items():
    # fit model
    v.fit(X_train, y_train.ravel())

    # predict y's
    pred = v.predict(X_test)

    # accuracy score
    sens = recall_score(y_test, pred, pos_label=1) # correctly identified survived.
    spec = recall_score(y_test, pred, pos_label=0) # correct identified died.

    # print results
    print(k + '; Sensitivity: ' + '{:.1%}'.format(sens) + '; Specificity: ' + '{:.1%}'.format(spec))

# Logistic Regression consistently offers very high specificity relative
# to the other models. Get a fitted model to use on the testing data.
model = LogisticRegression()
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)
print('Sensitivity: {:.1%}'.format(recall_score(y_test, y_pred, pos_label=1)))
print('Specificity: {:.1%}'.format(recall_score(y_test, y_pred, pos_label=0)))

# %% Run on testing data

# read in testing data
test_data = pd.read_csv('data/test.csv')

# clean up
test_data, X_testing = clean_data(test_data, training=False)

# predict using fitted logit model from above
y_pred = model.predict(X_testing)

# save final dataframe
final_predictions = pd.DataFrame({'PassengerId': test_data.index,
                                  'Survived': y_pred})
final_predictions.to_csv('submission.csv', index=False, header=True)

# uploaded to kaggle via API:
# kaggle competitions submit -c titanic -f submission.csv -m "Message"
