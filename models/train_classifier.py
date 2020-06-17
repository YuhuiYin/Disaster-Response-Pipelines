import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(database_filepath):
    engine = create_engine('sqlite:///./{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterTable', engine)
    category_names = ['related', 'request', 'offer',
                      'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                      'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                      'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                      'infrastructure_related', 'transport', 'buildings', 'electricity',
                      'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                      'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                      'other_weather', 'direct_report']
    X = df['message']
    Y = df[category_names]

    return X,Y,category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=36))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    X_train, X_test, y_train, y_test = train_test_split(X_test, Y_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test = y_test.to_numpy()

    for i in range(0, 35):
        print('Classification Report of {} :\n {}' \
              .format(category_names[i], classification_report(y_test[:, i], y_pred[:, i])))


def save_model(model, model_filepath):
    pickle.dump(model, open('./{}'.format(model_filepath), 'wb'))


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