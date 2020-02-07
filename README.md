Disaster Response Pipeline Project

This project is a message classifier for disaster response by machine learning. Messages will be classified to 36 catagroies.



1. Content

   1.Data: 
         1.two dataset(disaster_categories.cvs and disaster_messages.cvs).

         2.one py file (process_data.py), reads in the data, cleans and stores it in a SQL database.

         3.one database file (DisasterResponse.db) from transformed and cleaned data.

    2.Model: 
         1.one py file(train_classifier.py), load data, transform it using natural language processing, run a machine learning model using            GridSearchCV.

    3.APP: 
         1.One py file(run.py), include Flask, user interface and display.
         2.Templates, A folder containing the html templates.


2. Instructions:

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database 

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves 

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. python run.py
Go to http://0.0.0.0:3001/
