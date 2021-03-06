#file for testing on test set

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#importing dataset
df = pd.read_csv('reviews.csv')
print(df.groupby('Liked').count())

#defining variables
X = df['Review']
y = df['Liked']

 # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=45)


#using count vectorizer
vect = CountVectorizer(stop_words='english')

vect_train= vect.fit_transform(X_train)
vect_test = vect.transform(X_test)

#converting sparse array to dense array
X_train_fit = vect_train.toarray()
X_test_fit = vect_test.toarray()


# Make predictions on test dataset using naive bayes classifier

#loading model from pickle
NB_model = pickle.load(open('KNNmodel.sav', 'rb'))
NB_model.fit(X_train_fit, y_train)
predictions = NB_model.predict(X_test_fit)


# Evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
