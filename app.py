from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utlis import preprocess_reviews

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib # save and load model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-model',
                    type=str,
                    default='trigram',
                    help='which model to be used for sentiment analysis')

parser.add_argument('-modelPath',
                    type=str,
                    default='model/bert',
                    help='the path of the model')

FLAGS, unparsed = parser.parse_known_args()

# load model of encoding and prediction
if FLAGS.model == 'trigram':
    clf = joblib.load("model/final_model_small.m")
    ngram_vectorizer = joblib.load("model/vectorizer_small.m")
elif FLAGS.model == 'bert':
    import tensorflow as tf
    from bert import run_classifier
    from utlis_bert import create_tokenizer_from_hub_module, serialize_example
    # Please make sure you use Tensorflow 1.X
    print('Tensorflow version: {}'.format(tf.__version__))
    label_list = [0, 1]
    MAX_SEQ_LENGTH = 128
    tokenizer = create_tokenizer_from_hub_module()

    # Load BERT model

    # latest model file path
    # from pathlib import Path
    # export_dir = 'model/bert'
    # subdirs = [x for x in Path(export_dir).iterdir()
    #            if x.is_dir() and 'temp' not in str(x)]
    # latest = str(sorted(subdirs)[-1])

    model_path = FLAGS.modelPath
    print("model_path: {}".format(model_path))
    from tensorflow.contrib import predictor
    predict_fn = predictor.from_saved_model(model_path)
    # print(predict_fn)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()

# This is the tiny database
class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    sentiment_polarity = db.Column(db.Integer)

    def __repr__(self):
        return '<Task %r>' % self.id

# Main api: input the reviews and get the sentiment
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        print('Task_Content : {}'.format(task_content))

        test = preprocess_reviews([task_content])
        print('review: {}'.format(test))
        if FLAGS.model == 'trigram':
            test = ngram_vectorizer.transform(test)
            sentiment = int(clf.predict(test)[0])
        elif FLAGS.model == 'bert':
            input_examples = [run_classifier.InputExample(guid="", text_a = test[0], text_b = None, label = 0)] # here, "" is just a dummy label
            input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
            model_input = serialize_example(input_features[0].input_ids, input_features[0].input_mask, input_features[0].segment_ids, [input_features[0].label_id])
            sentiment = int(predict_fn({'example': [model_input]})['labels'])
        print('Sentiment: {}'.format(sentiment))
        print(type(sentiment))

        new_task = Todo(content=task_content, sentiment_polarity = sentiment)


        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your review'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html', tasks=tasks)

# Delete the review
@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that review'

# Update the review in the database
@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        test = preprocess_reviews([request.form['content']])
        test = ngram_vectorizer.transform(test)
        sentiment = int(clf.predict(test)[0])

        task.sentiment_polarity = sentiment

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your review'

    else:
        return render_template('update.html', task=task)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
