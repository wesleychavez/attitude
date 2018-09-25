from flask import Flask, request, render_template
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from gensim.models.word2vec import Word2Vec


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text_input']
    result = text.upper()

    phrase = text.lower().split(' ')
    word2vec = Word2Vec.load("word2vec.model")
    def word2idx(word):
        return word2vec.wv.vocab[word].index

    x = np.zeros([1, 56], dtype=np.int32)
    for i, word in enumerate(phrase):
        x[0,i] = word2idx(word)

    # load json and create model
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    prediction = model.predict_classes(x)
    pred = str(prediction[0])
    return render_template("result.html",result = prediction, text_input = text)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
