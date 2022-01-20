#import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Load the dataset
df = pd.read_csv("HomeGardenCrops.csv")

df = df.iloc[:,:].values
df1 = pd.DataFrame(df)

#Get first 5 rows in the dataset
df1.head()

#Split the dataset into feature and target variables
x = df1.iloc[:,0:5]
y = df1.iloc[:, [-2,-1]]


print(x)
print(y)


#Split the features and target variable of the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

#Display train and test values

print(x_train)
print(y_test)
print(x_test)


#Build and train the DecisionTreeClassifier model

clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

Predict = clf.predict(x_test)

print(Predict)

import  pickle

pickle.dump(clf, open('crop.model', 'wb'))





