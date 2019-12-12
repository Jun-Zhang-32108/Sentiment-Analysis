# Sentiment-Analysis on Movie Review Data 
## Author: Jun Zhang \<jun.1.zhang@aalto.fi>

To run the program: 

        python3 app.py

To install the dependencies:

        pip3 install -r requirements.txt

A live demo you can try out: 

        http://52.156.250.103:5000/

The overview of the project can be found:
https://docs.google.com/presentation/d/1mENl24uh39z9Ett99aFVKVXii-qRruEmsRaLOm6jhvk/edit?usp=sharing.

Directory  Structures:

app.py - main application

utlis.py - preprocessing function

test.db - SQLAlchemy database

data - three datasets used in the projects, MDB dataset (movie_data), Rotten Tomato(rottenTomatoes) dataset from Kaggle and one dataset from the company(movie_review_data).

model - pretrained models used for feature extraction and prediction

jupyter_notebook: sourcecode for experiments and training.

env - virtual environment

static, templates: source code for the webpage