import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
import joblib
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
   """
   Function to load the dataset
   Input: Databased filepath
   Output: Returns the Features X and target y along with target columns names catgeory_names
   """
   table_name = 'table_one'
   engine = create_engine('sqlite:///{}'.format(database_filepath))
   df = pd.read_sql_table(table_name, engine)  
   category_names = df.columns[4:]
   X = df["message"].values               #Feature Matrix
   y = df[category_names].values           #Target Variable
   return X,y,category_names

    
def tokenize(text):
    '''
    Function to tokenize the text messages
    Input: message text
    output: cleaned tokenized text as a list object
    ''' 
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens        
        
        
        

def build_model():
   """
   input: None
   output: cv Grid search model object
   """

   pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

   parameters = {
        'clf__estimator__n_estimators': [20, 50]
        }

   cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3,n_jobs=-1, verbose=3)
   return cv


def evaluate_model(model, X_test, Y_test, category_names):
   """ 
   Fucntion for prints multi-output classification reports
   input:
   model: the scikit-learn fitted model
        X_text: The X test set
        Y_test: the Y test classifications
        category_names: the category names
    Returns:
        None
    """
   Y_pred = model.predict(X_test)
   for i in range(0,len(Y_pred.T)-1):
     print(classification_report(Y_test.T[i], Y_pred.T[i]))


def save_model(model, model_filepath):
   """
   Fucntion for save as pickle file
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