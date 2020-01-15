
#Getting input from user and predicting it
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from trainNB import vect

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


filename1 = 'NBmodel.sav'
    
loaded_model = pickle.load(open(filename1, 'rb'))

predict_me = input("Enter a Sentence: ")
dfObj = pd.DataFrame(columns=['Review'])

dfObj = dfObj.append({'Review': predict_me}, ignore_index=True)

fitted = vect.transform(dfObj['Review'])
check = fitted.toarray()
result = loaded_model.predict(check)
print(result)