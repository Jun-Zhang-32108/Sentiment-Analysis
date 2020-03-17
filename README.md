# Sentiment Analysis on Movie Review Data
### Author: Jun Zhang \<jun.1.zhang@aalto.fi>

**\# Updated on Feb 17th, 2020 - Add the support of GPT-style Transformer model**  

This repos is mainly about a simple RESTful api for sentiment analysis on movie review data. The api is built with **Flask**. Now the simple website support three different models: trigram+SVM , a [BERT](https://github.com/google-research/bert) based model which is fine-tuned on the movie review data and a [GPT-style](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) transformer with fine-tuning. The trigram + SVM scores around 90% ACC and F1 but performs just so so on the short reviews and not well on the negation while the BERT based model scores mores than 93% on ACC and F1 and performs quite well on the short especially emotional reviews. The GPT-style transformer model scores 92% on ACC and F1. The later two models both perform quite well on the ambiguous cases but BERT-based model is slightly better. A test example for BERT-based model and GPT-style Transformer can be seen from Figure 1 and Figure 2. A reasonable idea behind this is that word-embedding models like BERT can capture the deep contextual meaning of each words within each sentences while n-gram models fail to do that.  



<figure class="half">
    <h3> Figure 1 </h3>
    <img src="https://github.com/Jun-Zhang-32108/Sentiment-Analysis/blob/master/IMG/Fig1.png" width="400" title="Figure1"/>
    <h3> Figure 2 </h3>
    <img src="https://github.com/Jun-Zhang-32108/Sentiment-Analysis/blob/master/IMG/Fig2.png" width="400" title="FIgure2"/>   
</figure>




To run the program with trigram + SVM model:

        python3 app.py

To run the program with BERT model:

        python3 app.py -model bert -modelPath #where you store bert model#

To run the program with GPT-style Transformer model:

        python3 app.py -model transformer -modelPath #where you store transformer model#

A ready-to-used model trained by me with BERT base uncased on movie review data can be found [here](https://drive.google.com/drive/folders/1Fb-bwNUewYckwTQVu3A0pwkAK2znVex8?usp=sharing).

Fine-tuned transformer model on IMDB dataset can be found [here](https://drive.google.com/open?id=1nRKgnsazET0N5TpTPHXLerNQNvkk-hdH)

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
 
 * `IMG` figures of test example


 ## Reference

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint [arXiv:1810.04805 (2018).](https://arxiv.org/abs/1810.04805)

Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. URL https://s3-us-west-2. amazonaws. com/openai-assets/researchcovers/languageunsupervised/language understanding paper. pdf, 2018.
