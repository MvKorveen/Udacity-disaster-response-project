# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('{0}'.format(database_filepath), engine) 

    # X-values
    X = df['message']

    # Get Y variables, i.e. all 36 categorical 
    category_names = list(df.columns[-36:])
    Y = df[category_names]

    return X, Y, category_names

def tokenize(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # POS tagging for lemmetizing
    # for reference: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    tagged_tokens = pos_tag(tokens)

    # Initialize lemmatizer and stopwords list
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    # List to hold cleaned and lemmatized tokens
    clean_tokens = []

    # Process each token with its POS tag
    for word, tag in tagged_tokens:
        # Remove non-alphabetic characters
        clean_word = re.sub(r"[^a-zA-Z]", " ", word).strip()

        # Skip the token if it's empty after cleaning
        if not clean_word:
            continue

        # Lemmatize based on the POS tag defined above, skip stopwords
        if clean_word.lower() not in stop_words:

            # Check if the token is a verb and apply lemmatization accordingly (pos='v')
            if tag.startswith('V'):  
                clean_tokens.append(lemmatizer.lemmatize(clean_word, pos='v').lower())
            # If not a verb
            else:  
                clean_tokens.append(lemmatizer.lemmatize(clean_word).lower())

    return clean_tokens

# Use message length as an input for the model
class MessageLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(msg) for msg in X]).reshape(-1, 1)

# Include the starting verb extractor from the udacity lesson "Case Study: Create Custom Transformer"
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        # Tokenize and POS tag the entire text at once
        pos_tags = pos_tag(tokenize(text))  # Tokenize the entire text
        if len(pos_tags) > 0:
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:  # Check if the first word is a verb
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
# Define a custom callback to count the grid search progress
def print_grid_search_progress(grid_search):
    total_combinations = np.prod([len(v) for v in grid_search.param_grid.values()])
    print(f"Total grid search combinations: {total_combinations}")
    
    def print_progress_callback(cv_results):
        # Print the progress after each iteration
        print(f"Grid search {cv_results['mean_test_score'].size}/{total_combinations} completed.")
    
    return print_progress_callback

def build_model():
    # Define feature extraction for text and custom features using FeatureUnion
    feature_union = FeatureUnion([
        ('text_features', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
            ('tfidf', TfidfTransformer())
        ])),

        ('msg_length', Pipeline([
            ('length', MessageLengthExtractor())
        ])),

        ('verb', Pipeline([
            ('verb_extractor', StartingVerbExtractor())
        ]))
    ])

    # Main pipeline with classifier
    pipeline = Pipeline([
        ('features', feature_union),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    # Asseses the number of features used by the TF-IDF vectorizer
    'features__text_features__vect__max_features': [5000, 7500],

    # Handle class imbalance, e.g. we have a lot of variables with no positive observations
    'clf__estimator__class_weight': ['balanced', {0: 1, 1: 3}],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)


    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # Predict on test data
    Y_pred = model.predict(X_test)

    # Loop through the number of target variables and check the scores for each variable
    for i, category in enumerate(category_names):
        print(f"Classification report for output {category}:")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i], zero_division=0))

def save_model(model, model_filepath):
    # Save the model
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

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