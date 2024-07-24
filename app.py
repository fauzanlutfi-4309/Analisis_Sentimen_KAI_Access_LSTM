# app.py
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import re
import nltk
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

MODEL_PATH = 'sentiment_analysis_model.h5'
loaded_model = load_model(MODEL_PATH)

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stop_words_indonesia = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
max_len = 50
# TAHAP CLEANING TEXT
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    emoji_pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text

    return text

def casefoldingText(text):
    text = text.lower()
    return text

def tokenizingText(text):
    text = word_tokenize(text)
    return text

def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word not in stop_words_indonesia]
    return filtered_tokens

def stem_text(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def preprocess_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    tokens = tokenizingText(text)
    tokens = remove_stopwords(tokens)
    stemmed_tokens = stem_text(tokens)
    return stemmed_tokens

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction_result="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        new_sentence = request.form['new_sentence']
        if new_sentence:
            preprocessed_sentence = preprocess_text(new_sentence)
            tokenizer.fit_on_texts([preprocessed_sentence])  # Menambahkan langkah ini
            new_sequence = tokenizer.texts_to_sequences([preprocessed_sentence])
            new_padded = pad_sequences(new_sequence, maxlen=max_len, padding='pre', truncating='post')
            prediction = loaded_model.predict(new_padded)
            sentiment_label = ['negative', 'neutral', 'positive']
            sentiment = sentiment_label[np.argmax(prediction)]
            return redirect(url_for('result', prediction_result=sentiment))
            
    return render_template('index.html', prediction_result="No text input.")

@app.route('/result/<prediction_result>')
def result(prediction_result):
    return render_template('result.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
