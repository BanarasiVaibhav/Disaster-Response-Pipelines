# import libraries
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

# import NLP libraries
import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization


# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier



# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
X = df['message']
Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

def tokenize(text):
    # Define url pattern regular expression
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls string.replace()
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]
    
    return clean_tokens



pipelineKNN = Pipeline([
    ('vectorizer',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultiOutputClassifier(KNeighborsClassifier()))
])


pipelineRF = Pipeline([
    ('vectorizer',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultiOutputClassifier(RandomForestClassifier()))
])


pipelineADA = Pipeline([
    ('vectorizer',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultiOutputClassifier(AdaBoostClassifier()))
])



X_train, X_test, y_train, y_test = train_test_split(X, Y)


pipelineKNN.fit(X_train,y_train) #K Nearest Neighbor classifir pipeline

pipelineRF.fit(X_train,y_train) #Random forest classifir pipeline

pipelineADA.fit(X_train,y_train) #Ada Boost Classifier pipeline


def build_report(pipeline, X_test, y_test):
    # predict on the X_test
    y_pred = pipeline.predict(X_test)
    
    # build classification report on every column
    performances = []
    for i in range(len(y_test.columns)):
        performances.append([f1_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             precision_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             recall_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                index = y_test.columns)   
    return performances


build_report(pipelineKNN, X_test, y_test)
build_report(pipelineRF, X_test, y_test)
build_report(pipelineADA, X_test, y_test)




parameters = {'tfidf__use_idf':[True,False],
              'classifier__estimator__learning_rate':[0.5,1],
             'classifier__estimator__n_estimators':[50, 100]}

print(pipelineADA.get_params().keys())

cv = GridSearchCV(estimator=pipelineADA, param_grid = parameters)

cv.fit(X_train, y_train)

cv.best_params_


build_report(cv, X_test, y_test)


pickle.dump(pipelineADA, open('adaboost_model.pkl', 'wb'))