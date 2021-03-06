##file for saved training models

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

#importing dataset
df = pd.read_csv('reviews.csv')
print(df.groupby('Liked').count())

#visualizing the dataset
df.hist(column='Liked')
plt.xlabel('Type of review')
plt.ylabel('No.of reviews')
plt.show()

#defining function for removing stopwords 
# def stopword(review):
#     lowerwords=review.lower()
#     words = re.split("\s+", lowerwords)
#     stop_words = set(stopwords.words('english'))
#     filterwords=[w for w in words if not w in stop_words]
#     return " ".join(f for f in filterwords)

# edit_reviews = []
# for i in range(0,len(df['Review'])):
#     edit_reviews.append(stopword(df['Review'][i]))


# #adding new column with reviews with removed stop words
# df['edit_reviews'] = edit_reviews

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

##Using the Knearest neighbour Algorithm
# -----------------------------------

print("USING DESICISION TREE CLASSIFIER")

CARTmodel = DecisionTreeClassifier()
CARTmodel.fit(X_train_fit, y_train)
#saving trained model in pickle
filename6 = 'CARTmodel.sav'
pickle.dump(CARTmodel, open(filename6, 'wb'))