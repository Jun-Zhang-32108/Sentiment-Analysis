# Sentiment Analysis on Movie Review Data
### Author: Jun Zhang \<jun.1.zhang@aalto.fi>

**\# Updated on Feb 22th, 2020**  

This repos is mainly about a simple RESTful api for sentiment analysis on movie review data. The api is built with **Flask**. Now the simple website support two different models: trigram+SVM and a [BERT](https://github.com/google-research/bert) based model which is fine-tuned on the movie review data. The trigram + SVM scores over 90% ACC and F1 but performs just so so on the short reviews and not well on the negation while the BERT based model scores only 88% on ACC and F1 but performs quite well on the short especially emotional reviews. A reasonable idea behind this is that word-embedding models like BERT can capture the deep contextual meaning of each words within each sentences while n-gram models fail to do that.  

To run the program with trigram + SVM model:

        python3 app.py

To run the program with BERT model:

        python3 app.py -model bert -modelPath #where you store bert model#

A ready-to-used model trained by me with BERT base uncased on movie review data can be found [here](https://drive.google.com/drive/folders/1Fb-bwNUewYckwTQVu3A0pwkAK2znVex8?usp=sharing).

To install the dependencies:

        pip3 install -r requirements.txt

A live [demo](http://52.156.250.103:5000/) you can try out [here](http://52.156.250.103:5000/).

The presentation of the project can be found [here](https://docs.google.com/presentation/d/1mENl24uh39z9Ett99aFVKVXii-qRruEmsRaLOm6jhvk/edit?usp=sharing).

## Contents

This repository is organized as follows:

 * `app.py` main application

 * `utlis.py` preprocessing functions

  * `utlis_bert.py` preprocessing functions for BERT model

 * `test.db` SQLAlchemy database

 * `data` three datasets used in the projects, MDB dataset (movie_data), Rotten Tomato(rottenTomatoes) dataset from Kaggle and one dataset from the company(movie_review_data).

 * `model` - pretrained models used for feature extraction and prediction

 * `jupyter_notebook` sourcecode for experiments on training.

 * `env` virtual environment

 * `static, templates` source code for the webpage


 ## Reference

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint [arXiv:1810.04805 (2018).](https://arxiv.org/abs/1810.04805)
