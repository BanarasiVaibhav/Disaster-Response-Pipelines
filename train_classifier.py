import sys
# basic libraries
import pandas as pd
import numpy as np
#SQL library
from sqlalchemy import create_engine
# NLP
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download(['punkt', 'wordnet','stopwords'])


# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
# pickle to store model
import pickle


def load_data(database_filepath):
    """
    Load disaster dateset from SQLite & split dataset into messages and categories 
    
    Input: database_filepath: path of SQLite database file of processed dataset
    Return: X: Message part of disaster_message dataframe
        y: category part of disaster_message dataframe
        categories: names of categories, columns of y
    """
    # Load SQLite dataset
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    """
    Clean and tokenize the message
    
    Input: text: the message for tokenization(X from load_data)
    Return: clean_tokens: token list of message
    """
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


def build_model():
    """
    Builds the pipeline of NLP + Classification, Tf-Idf as message 
    transformation and AdaBoost as 
    classification model, along with GridSearchCV strategy
    
    Return: model: machine learning model described above
    """
    #pipelineKNN = Pipeline([
    #('vectorizer',CountVectorizer(tokenizer=tokenize)),
    #('tfidf',TfidfTransformer()),
    #('classifier',MultiOutputClassifier(KNeighborsClassifier()))
    #])


    #pipelineRF = Pipeline([
    #('vectorizer',CountVectorizer(tokenizer=tokenize)),
    #('tfidf',TfidfTransformer()),
    #('classifier',MultiOutputClassifier(RandomForestClassifier()))
    #])

    # pipeline containing Tf-Idf and AdaBoost
    pipelineADA = Pipeline([
    ('vectorizer',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultiOutputClassifier(AdaBoostClassifier()))
    ])   

    # parameter set for GridSearchCV
    parameters = {'tfidf__use_idf':[True,False],
              'classifier__estimator__learning_rate':[0.5,1],
             'classifier__estimator__n_estimators':[50, 100]}

    # Instantiate GridSearchCV
    cv = GridSearchCV(estimator=pipelineADA, param_grid = parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance and print metrics of validation
    
    Input:model: classification model from build_model()
        X_test: test set of X(messages)
        Y_test: test set of y(categories)
        category_names: names of message categories(categories column names)
    """
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



def save_model(model, model_filepath):
    """
    Save classification model to pickle file
    
    Input:model: validated classification model
        model_filepath: specified storage path
    """
    pickle.dump(pipelineADA, open('adaboost_model.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()