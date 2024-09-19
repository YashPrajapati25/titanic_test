

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#train_data = pd.read_csv(r'train.csv')
test_data = pd.read_csv(r'test.csv')
gender_submission = pd.read_csv(r'gender_submission.csv')

test_data.head()

test_data.describe()

test_data.columns

test_data.dtypes

test_data.isnull().sum()
column_names = test_data.columns
for column in column_names:
    print(column + ' - ' + str(test_data[column].isnull().sum()))

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

test_data.head()

test_data = test_data.drop(columns=['Ticket', 'Cabin'])

test_data.head()

test_data['Sex']=test_data['Sex'].map({'male':0,'female':1})
test_data['Embarked']=test_data['Embarked'].map({'S':2,'C':0,'Q':1})

test_data.head()

test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data = test_data.drop(columns='Name')
test_data.Title.unique()

test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

plt = test_data.Title.value_counts().sort_index().plot(kind='bar')
plt.set_xlabel('Title')
plt.set_ylabel('Passenger count')

test_data['Title'] = test_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})

test_data.head()

corr_matrix = test_data.corr()

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 8))
sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)

test_data.isnull().sum()

fare_mean=test_data['Fare'].mean()
test_data['Fare'] = test_data['Fare'].fillna(fare_mean)
test_data.isnull().sum()

age_median_test = test_data.Age.median()
test_data.Age = test_data.Age.fillna(age_median_test)
print(age_median_test)

test_data.Title = test_data.Title.fillna(2)

test_data.isnull().sum()

X_test= test_data.drop(columns='PassengerId')