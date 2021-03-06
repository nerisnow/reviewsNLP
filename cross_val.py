##file for cross validation

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


#importing dataset
df = pd.read_csv('reviews.csv')
print(df.groupby('Liked').count())

#visualizing the dataset
df.hist(column='Liked')
plt.title('Histogram of total reviews in dataset')
plt.xlabel('Type of review')
plt.ylabel('No.of reviews')
plt.show()

#defining variables
X = df['Review']
y = df['Liked']

 # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=45)

#visualizing the train dataset
y_train.hist()
plt.title('Histogram of train set reviews')
plt.xlabel('Type of review')
plt.ylabel('No.of reviews')
plt.show()

#visualizing the test dataset
y_test.hist()
plt.title('Histogram of test set reviews')
plt.xlabel('Type of review')
plt.ylabel('No.of reviews')
plt.show()


#count vectroizer for reviews
vect = CountVectorizer().fit_transform(X_train)
X_train_fit = vect.toarray()

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', MultinomialNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RFC',RandomForestClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train_fit, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: Average Accuracy = %f' % (name, cv_results.mean()))