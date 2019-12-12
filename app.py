from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utlis import preprocess_reviews

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib # save and load model

# load model of encoding and prediction
clf = joblib.load("model/final_model_small.m")
ngram_vectorizer = joblib.load("model/vectorizer_small.m")

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
        test = ngram_vectorizer.transform(test)
        sentiment = int(clf.predict(test)[0])
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
