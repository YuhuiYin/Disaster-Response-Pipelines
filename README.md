# Disaster-Response-Pipelines

## Installation
The following libraries should be installed:
json, plotly, nltk, flask, sklearn, sys, pickle, re and sqlalchemy.

## Project Motivation
It is important for emergency workers to categorize messages so that they can send the messages to an appropriate disaster relief agency. To make it esier, I have created a machine learning pipeline as well as a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions
There are three folders available here to showcase work related to this project:
1.**data** : There are two oringinal data documents, one database path and an ETL pipeline that cleans data and stores in database.
2.**models** : There is a ML pipeline that trains classifier and saves.
3.**app** : There are two HTML documents and one python document to run the Web App.

## Instructions for running files
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
