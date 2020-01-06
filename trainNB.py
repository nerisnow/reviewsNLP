import pandas as pd
import numpy as np
import pickle
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

#importing dataset
df = pd.read_csv('reviews.csv')

#defining function for removing stopwords 
def stopword(review):
    lowerwords=review.lower()
    words = re.split("\s+", lowerwords)
    stop_words = set(stopwords.words('english'))
    filterwords=[w for w in words if not w in stop_words]
    return " ".join(f for f in filterwords)

edit_reviews = []
for i in range(0,len(df['Review'])):
    edit_reviews.append(stopword(df['Review'][i]))
#print(edit_reviews)


#adding new column with reviews with removed stop words
df['edit_reviews'] = edit_reviews

#defining variables
X = df['edit_reviews']
y = df['Liked']

 # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=45)


##Using the Naive Bayes Algorithm
# -----------------------------------

print("USING NAVIE BAYES")
#forming pipeline with vectorizer and model
pipelineNB = Pipeline([('vectorizer', CountVectorizer()),
                    ('modelNB', naive_bayes.MultinomialNB())])

#training model
trainmodel = pipelineNB.fit(X_train, y_train)

#cross validation
scores = cross_val_score(pipelineNB, X_train, y_train, cv=10)
print(scores)
print("Accuracy: %0.2f " % (scores.mean()))

#saving trained model in pickle
filename1 = 'NBmodel.sav'
pickle.dump(trainmodel, open(filename1, 'wb'))
# pipelineNB.fit_transform(X, y)  

#loading model from pickle
loaded_model = pickle.load(open(filename1, 'rb'))

#testing model with saved model
testmodel = loaded_model.predict(X_test)

#evaluating the model
accuracy_nb = accuracy_score(y_test, testmodel)
print("Test Accuracy using Navies Bayes: {:.1f}%".format(accuracy_nb*100))



##Using the Random Forest Algorithm
# -----------------------------------

print("USING RANDOM FOREST")
from sklearn.ensemble import RandomForestClassifier
#forming pipeline using the random forest
pipelineRf = Pipeline([('vectorizer', CountVectorizer()),
                    ('modelRF', RandomForestClassifier())])

#training model
trainmodelrf = pipelineRf.fit(X_train, y_train)

#cross Validation
scoresrf = cross_val_score(pipelineRf, X_train, y_train, cv=10)
print(scoresrf)
print("Accuracy: %0.2f " % (scoresrf.mean()))


#saving trained model in pickle

filenamerf = 'randomforestmodel.sav'
pickle.dump(trainmodelrf, open(filenamerf, 'wb'))

#testing model with saved model

loaded_rf_model = pickle.load(open(filenamerf, 'rb'))

trainmodelrf = loaded_rf_model.predict(X_test)
accuracyrf = accuracy_score(y_test, trainmodelrf)
print("Test Accuracy Using Random Forest: {:.1f}%".format(accuracyrf*100))



##Using the K-Nearest Algorithm
# -----------------------------------

print("USING K-NEAREST ALGORITHM")
from sklearn.neighbors import NearestCentroid

pipeline_near = Pipeline([('vectorizer', CountVectorizer()),
                    ('modelRF', NearestCentroid())])

trainmodel_knear = pipeline_near.fit(X_train, y_train)

score_near= cross_val_score(trainmodel_knear, X_train, y_train, cv=10)
print(score_near)
print("Accuracy: %0.2f " % (score_near.mean()))

k_meansfile = 'k_means.sav'
pickle.dump(trainmodel_knear, open(k_meansfile, 'wb'))
loaded_knear = pickle.load(open(k_meansfile, 'rb'))
trainmodel_knear = loaded_knear.predict(X_test)
accuracy_knear = accuracy_score(y_test, trainmodel_knear)
print("Test Accuracy using k nerest: {:.2f}%".format(accuracy_knear*100))




##Using the Logistic Algorithm
# -----------------------------------

print("USING LOGISTIC REGRESSION ALGORITHM")

from sklearn.linear_model import LogisticRegression
pipeline_logis = Pipeline([('vectorizer', CountVectorizer()),
                    ('modelRF', LogisticRegression(random_state=0))])

trainmodel_logis = pipeline_logis.fit(X_train, y_train)

score_logis= cross_val_score(trainmodel_logis, X_train, y_train, cv=10)
print(score_logis)
print("Accuracy: %0.2f " % (score_logis.mean()))

logis = 'logis.sav'
pickle.dump(trainmodel_logis, open(logis, 'wb'))
loaded_logis = pickle.load(open(logis, 'rb'))

trainmodel_logis = loaded_logis.predict(X_test)

accuracy_logis = accuracy_score(y_test, trainmodel_logis)
print("Test Accuracy using Logistic regression : {:.2f}%".format(accuracy_logis*100))