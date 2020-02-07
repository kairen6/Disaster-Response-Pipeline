# Disaster Response Pipeline Project
This project is a message classifier for disaster response by machine learning. Messages will be classified to 36 catagroies.

Content

Data:
two dataset(disaster_categories.cvs and disaster_messages.cvs),
one py file (process_data.py), reads in the data, cleans and stores it in a SQL database.
one database file (DisasterResponse.db) from transformed and cleaned data.

Model:
one py file(train_classifier.py), load data, transform it using natural language processing, run a machine learning model using GridSearchCV.

APP:
one py file(run.py), include Flask, user interface and display.

templates, A folder containing the html templates

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



