from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

## Definitions
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

app=Flask(__name__)

df = pd.read_csv("sentiment.tsv",sep = '\t')
df.columns = ["label","body_text"]
# Features and Labels
df['label'] = df['label'].map({'pos': 0, 'neg': 1})
df['tweet'] = np.vectorize(remove_pattern)(df['body_text'],"@[\w]*")
tokenized_tweet = df['tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
df['tweet'] = tokenized_tweet
df['body_len'] = df['body_text'].apply(lambda x:len(x) - x.count(" "))
df['punct%'] = df['body_text'].apply(lambda x:count_punct(x))
X = df['tweet']
y = df['label']
print(type(X))
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the df
X = pd.concat([df['body_len'],df['punct%'],pd.DataFrame(X.toarray())],axis = 1)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
## Using Classifier
clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=2000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y)

@app.route('/')
def index():
	return "hello Flask"

if __name__ == '__main__':
	app.run(debug=True)

