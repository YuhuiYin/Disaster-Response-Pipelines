import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load modelcd
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Graph one
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # create visuals
    graph_one = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]



    # Graph two
    # extract data
    category_names = ['related', 'request', 'offer',
                      'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                      'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                      'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                      'infrastructure_related', 'transport', 'buildings', 'electricity',
                      'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                      'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                      'other_weather', 'direct_report']
    category_counts = df[category_names].sum()

    # create visuals
    graph_two = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Occurrence of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # Graph three
    # extract data
    related_category = ['request', 'offer',
                        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                        'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                        'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                        'infrastructure_related', 'transport', 'buildings', 'electricity',
                        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                        'other_weather', 'direct_report']
    df_related = df[df['related'] == 1]
    rate = []
    for val in related_category:
        rate.append(df_related[val].sum() / len(df_related))
    # create visuals
    graph_three = [
        {
            'data': [
                Bar(
                    x=related_category,
                    y=rate
                )
            ],

            'layout': {
                'title': 'Occurrence Rates of Related Categories',
                'yaxis': {
                    'title': "Rate"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    graphs = []
    graphs.append(graph_one)
    graphs.append(graph_two)
    graphs.append(graph_three)

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
    classification_results = dict(zip(df.columns[4:], classification_labels))

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