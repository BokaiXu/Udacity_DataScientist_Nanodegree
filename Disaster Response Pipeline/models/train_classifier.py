import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_data(database_filepath):
    """
    This is a function to load data from database.
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterRecord',engine)
    X = df['message'].values
    y = df.drop(['id','message','genre'],axis=1)

    return X,y.values, np.array(y.columns)

def tokenize(text):
    """
    This is the function used to convert text to a list of words.
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """
    This is the function used to build the machine learning pipeline.
    """
    pipeline=Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Grid Search
    parameters = {'vect__min_df': [1, 5],
                             'tfidf__use_idf':[True, False],
                             'clf__estimator__n_estimators':[10, 25], 
                             'clf__estimator__min_samples_split':[2, 5, 10]}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This is a function to evaluate the model.
    """
    y_pred=model.predict(X_test)
    for i in range(36):
        print(category_names[i])
        print('accuracy_score ',accuracy_score(Y_test.T[i],y_pred.T[i]))
        print('classification_report \n', classification_report(Y_test.T[i],y_pred.T[i]))

def save_model(model, model_filepath):
    """
    This function saves the trained model.
    """

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
