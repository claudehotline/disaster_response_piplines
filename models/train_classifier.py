import sys
import pickle

import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''
    Load the cleaned dataset from the sqlite database specified
    by database_filepath.

    input:
        database_filepath: The path of the sqlite database file.

    return:
        X: The message column of the dataset.
        Y: All categories columns of the dataset.
        categories: the column names of all categories columns.
    '''
    # load data from database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('ResponseCategory', engine)
    df = df.drop(df.loc[df['related'] > 1, :].index, axis=0)
    categories = df.columns[-36:]
    X = df['message'].values
    Y = df[categories]

    return X, Y, categories


def tokenize(text):
    '''
    Convert given text into tokens.

    input:
        text: The text to be tokenized.

    return:
        clean_tokens: the tokens of the input text.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Construct a scikit-learn pipeline and use GridSearchCV method to
    tune the pipelines hyperparameters.

    reutrn:
        model: The scikit-learn pipeline model.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters,
                      verbose=200, return_train_score=False, n_jobs=20)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Use model to perform predictions

    input:
        model: Model using to perform predictions.
        X_test: Test messages.
        Y_test: True values of the categories for corresponding messages.
        category_names: the name of each category
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names= category_names))


def save_model(model, model_filepath):
    '''
    Save the model in a pickle file.

    input:
        model: Model to be saved.
        model_filepath: the file path of the saved model.
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
