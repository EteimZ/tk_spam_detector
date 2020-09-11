#Importing all neccessary libraries
import numpy as np
import pandas as pd
import string
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Loading the data
df = pd.read_csv('spam.csv',
             usecols = [0,1], encoding = 'ISO-8859-1')
df.rename(columns = {'v1': 'Category','v2': 'Message'},inplace = True)

df['label_num'] = df.Category.map({'ham':0, 'spam':1})

def text_process(mess):
    """
    This function is used to process the text by removing all stop words 
    and punctuations
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

df['clean_msg'] = df.Message.apply(text_process) #Applies text_process to each message.

X = df.clean_msg
y = df.label_num

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Vectorizing the text data.
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

#Using Naive Bayes model.
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

#Saving the model
filename = 'spam.sav'
joblib.dump(nb, filename)
