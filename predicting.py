
#GettiNG AND PREDECTIONG ReSULT
import pandas as pd
import numpy as np
import pickle
import re

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


filename1 = 'NBmodel.sav'
    
loaded_model = pickle.load(open(filename1, 'rb'))

predict_me = input("Enter a Sentence: ")
dfObj = pd.DataFrame(columns=['Review'])

dfObj = dfObj.append({'Review': predict_me}, ignore_index=True)

def stopword(review):
    lowerwords=review.lower()
    words = re.split("\s+", lowerwords)
    stop_words = set(stopwords.words('english'))
    filterwords=[w for w in words if not w in stop_words]
    return " ".join(f for f in filterwords)

edit_opinion = []
for i in range(0,len(dfObj['Review'])):
    edit_opinion.append(stopword(dfObj['Review'][i]))
print(edit_opinion)


dfObj['edit_opinion'] = edit_opinion
check = dfObj["edit_opinion"]
result = loaded_model.predict(check)
print(result)