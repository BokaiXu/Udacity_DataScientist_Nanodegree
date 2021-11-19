Libraries needed:
Pandas
SQLAlchemy
Sklearn
Nltk
Re
Plotly

Summary:
In this project, a data set containing real messages that were went during disaster events were used to create a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.
It includes a web app where an emergency worker can input a new message and get classification results in several categories. This web app also displays visualizations of the data.

How to run the python script and the web app:
Python script: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
               python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Web app: python run.py
         Open a new terminal
         env|grep WORK
         In a new web browser: https://SPACEID-3001.SPACEDOMAIN

File structure of the project:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db  # database to save clean data to
|- DisasterRecord.db  #data table to load in the app

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb
- categories.csv
- DisasterRecord.db
- DisasterResponse.db
- ETL Pipeline Preparation.html
- ML Pipeline Preparation.html
- messages.csv

Results:
Random forest algorithm was used to build the model to classify data. A grid search was used to optimize parameters.
The raw data was firstly treated to a list of words, then the model classify the data to 36 categories.
The model might be further improved with training more data.



Acknowledgement:
Udacity Inc.
