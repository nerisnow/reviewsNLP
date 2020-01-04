from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import pickle
import re



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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=45) # 70% training and 30% test

#using count vectorizer method to count number of unique words
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(X_train)
print ("Number of unique words in the train dataset: {}".format(
    len(count_vectorizer.get_feature_names())
))


preprocessing_pipeline = Pipeline([
    ("Count Vectorization", CountVectorizer()),
])
filtered_features = preprocessing_pipeline.fit_transform(X_train)

#training model
nb = naive_bayes.MultinomialNB()
nb.fit(filtered_features, y_train)


# filename1 = 'NBmodel.sav'
# pickle.dump(nb, open(filename1, 'wb'))

# filename2 = 'vectorizer.sav'
# pickle.dump(count_vectorizer, open(filename2, 'wb'))

#testing model
# loaded_model1 = pickle.load(open(filename1, 'rb'))
# loaded_model2 = pickle.load(open(filename2, 'rb'))

# countstest = loaded_model2.fit_transform(X_test)
# print ("Number of unique words in the ttest dataset: {}".format(
#     len(count_vectorizer.get_feature_names())
# ))
# test_predictions = loaded_model1.predict(countstest)



test_features = preprocessing_pipeline.transform(X_test)
test_predictions = nb.predict(test_features)

accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy: {:.1f}%".format(accuracy*100))