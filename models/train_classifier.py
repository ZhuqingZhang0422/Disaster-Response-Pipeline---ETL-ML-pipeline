# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier

import pickle


def load_data(database_filepath):
    #load data from 'ETL_pipeline_project'
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_disaster',engine)
    #df = df[:100]
    X = df['message']
    #Remove the unrelated columns
    Y = df.iloc[:,4:]
    category_names = list(df[:0])[4:]
    return X , Y , category_names 


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    words = word_tokenize(text)
    # remove stop words
    tokens = [w for w in words if w not in stopwords.words("english")]
    # lemmatize and stemming
    lemmatize = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    stemmed = [PorterStemmer().stem(w) for w in lemmatize]
    return stemmed


def build_model():
    pipeline = Pipeline([
    # Estimater
    ('vect', CountVectorizer(tokenizer=tokenize)),
    # Transformer
    ('tfidf',TfidfTransformer()),
    # Predictor and apply MultiOutputClassifier to assist computation
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
])
    # train classifier and set parameters for gridsearch
    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [4, 8]} 
    model = GridSearchCV(pipeline, param_grid = parameters)
    return model
    

def evaluate_model(model,X_test, Y_test, categories_names):
    '''
    Evaluation report of the model, and print the F1 scores
    Args:
        Input: model, x_test and y_test
        Output: evaluation report in text files 
    '''
    y_predict = model.predict(X_test)
    for index, col in enumerate(Y_test):
        print(categories_names[index])
        print(classification_report(Y_test[col], y_predict[:, index]))

def save_model(model, model_filepath):
    # save ML model to a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()