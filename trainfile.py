##file for saved training models

import pandas as pd
import pickle
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#importing dataset
df = pd.read_csv('reviews.csv')
print(df.groupby('Liked').count())

#defining variables
X = df['Review']
y = df['Liked']

 # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=45)

vect = CountVectorizer(stop_words='english')
vect_train= vect.fit_transform(X_train)
X_train_fit = vect_train.toarray()


##Using the Naive Bayes Algorithm
# -----------------------------------

print("USING NAVIE BAYES")

NBmodel = MultinomialNB()
NBmodel.fit(X_train_fit, y_train)
#saving trained model in pickle
filename1 = 'NBmodel.sav'
pickle.dump(NBmodel, open(filename1, 'wb'))


##Using the Logistic Regression Algorithm
# -----------------------------------

print("USING LOGISTIC REGRESSION")

LRmodel = LogisticRegression()
LRmodel.fit(X_train_fit, y_train)
#saving trained model in pickle
filename2 = 'LRmodel.sav'
pickle.dump(LRmodel, open(filename2, 'wb'))


##Using the SVM Algorithm
# -----------------------------------

print("USING SUPPORT VECTOR MACHINES")

SVMmodel = SVC(gamma='auto')
SVMmodel.fit(X_train_fit, y_train)
#saving trained model in pickle
filename3 = 'SVMmodel.sav'
pickle.dump(SVMmodel, open(filename3, 'wb'))

##Using the Random Forest Classifier
# -----------------------------------

print("USING RANDOM FOREST CLASSIFIER")

RFCmodel = RandomForestClassifier()
RFCmodel.fit(X_train_fit, y_train)
#saving trained model in pickle
filename4 = 'RFCmodel.sav'
pickle.dump(RFCmodel, open(filename4, 'wb'))

##Using the Knearest neighbour Algorithm
# -----------------------------------

print("USING K NEAREST NEIGHBOUR")

KNNmodel = KNeighborsClassifier()
KNNmodel.fit(X_train_fit, y_train)
#saving trained model in pickle
filename5 = 'KNNmodel.sav'
pickle.dump(KNNmodel, open(filename5, 'wb'))
