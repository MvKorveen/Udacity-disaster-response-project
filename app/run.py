#%%
import json
import plotly
import pandas as pd
import os
import numpy as np
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from langdetect import detect
import langcodes

#%%
app = Flask(__name__)

# redefine all functions used by the model

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

        # Skip the token if it's empty after cleaning to remove from output
        if not clean_word:
            continue

        # Lemmatize based on the POS tag defined above, skip stopwords
        if clean_word.lower() not in stop_words:

            # check if the token is a verb and apply tokenize accordingly (pos='v')
            if tag.startswith('V'):  
                clean_tokens.append(lemmatizer.lemmatize(clean_word, pos='v').lower())
            # if not a verb
            else:  
                clean_tokens.append(lemmatizer.lemmatize(clean_word).lower())

    return clean_tokens

# Use message lenght as an input for model
class MessageLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(msg) for msg in X]).reshape(-1, 1)
    

# Include the starting verb extractor from the udacity lesson "Case Study: Create Custom Transformer", 
# but removing tweet function
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):

        # Tokenize and POS tag the entire text at once
        pos_tags = pos_tag(tokenize(text))  # Tokenize the entire text once
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


#%%
# load data

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# get to data by way of the main folder (..) then to 'models'
db_path = os.path.join(script_dir, '..', 'data', 'DisasterResponse.db')

# Create the engine using the absolute path
engine = create_engine(f'sqlite:///{os.path.abspath(db_path)}')

df = pd.read_sql_table('data/DisasterResponse.db', engine)

#%%
# load model

# get to models by way of the main folder (..) then to 'models'
model_path = os.path.join(script_dir, '..', 'models', 'classifier.pkl')

model = joblib.load(model_path)

#%%
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # Get the overall count
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    
    # Get direct messages and count by language
    languange_counts = df[df['genre']=='direct'].groupby('language').count()['message'].sort_values(ascending=False)
    direct_message_count=languange_counts.sum()
    languange_share = languange_counts/direct_message_count
    languange_names = list(languange_counts.index)

    # Get the share of categories

    ## Get the set of Y categories
    category_names = list(df.columns[-36:])
    Y = df[df['genre']=='direct'][category_names]
    
    ## Initialize lists to store x and y values
    categories = []
    categories_share = []

    ## Loop through each column in the categorues to calculate the mean value (share chosen)
    for col in Y.columns:
        category_share = Y[col].mean()
        category = col
        categories.append(category)
        categories_share.append(category_share)

    # Create the visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    name='Number of Messages',
                    text=genre_counts,
                    textposition='outside',
                    marker=dict(color=['darkred' if genre == 'direct' else 'darkblue' for genre in genre_names])
                )
            ],

            'layout': {
                'title': {
                    'text': 'Number of messages per message type',
                    'font': {
                        'size': 20
                        }
                },
                'yaxis': {
                    'title': "Number of messages",
                    'range': [0, max(genre_counts)+2000]
                },
                'xaxis': {
                    'title': "Type of message",
                    'tickangle': 45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=languange_names,
                    y=languange_share,
                    name='Share of direct messages per language',
                    text=[round(val, 2) for val in languange_share],
                    textposition='outside',
                    marker=dict(color='darkred')
                )
            ],

            'layout': {
                'title': {
                    'text': 'Share of direct messages per language',
                    'font': {
                        'size': 20
                        }
                },
                'yaxis': {
                    'title': "Share of direct messages",
                    'range': [0, round(max(languange_share)+0.1, 1)]
                },
                'xaxis': {
                    'title': "Language of message",
                    'tickangle': 45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=categories_share,
                    name='Share of direct messages classified by category',
                    text=[round(val, 2) for val in categories_share],
                    textposition='outside',
                    marker=dict(color='darkred')
                )
            ],

            'layout': {
                'title': {
                    'text': 'Share of direct messages classified by category',
                    'font': {
                        'size': 20
                        }
                },
                'yaxis': {
                    'title': "Share of direct messages classified",
                    'range': [0, round(max(categories_share)+0.1, 1)]
                },
                'xaxis': {
                    'title': "Classification categories",
                    'tickangle': 45
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    # Change data collection since I added a column
    classification_results = dict(zip(df.columns[-36:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()