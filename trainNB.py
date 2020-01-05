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
accuracy = accuracy_score(y_test, testmodel)
print("Test Accuracy: {:.1f}%".format(accuracy*100))
