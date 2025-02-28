from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from gensim.models.word2vec import Word2Vec

app = Flask(__name__)

global graph
graph = tf.get_default_graph()
model = load_model('attitudemodel.h5')

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def my_form_post():
    
    # Make text input look like training dataset
    text = request.form['text_input']
    text = text.replace("'", " '")
    text = text.replace(",", " ,")
    text = text.replace(".", " .")
    text = text.replace("?", " ?")
    text = text.replace("!", " !")

    # To array
    phrase = text.lower().split(' ')

    # Load w2v model that we trained
    word2vec = Word2Vec.load("word2vec.model")
    def word2idx(word):
        return word2vec.wv.vocab[word].index

    # Replace out-of-vocab words with 'the'
    for i, word in enumerate(phrase):
        if word not in word2vec.wv.vocab:
            phrase[i] = 'the'

    # Indices instead of words... Embedding layer will take care of it
    x = np.zeros([1, 56], dtype=np.int32)
    for i, word in enumerate(phrase):
        x[0,i] = word2idx(word)

    # Predict and render
    with graph.as_default():
        prediction = model.predict_classes(x)
    pred = str(prediction[0]+1)
    return render_template("result.html", result=pred, text_input=text)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
