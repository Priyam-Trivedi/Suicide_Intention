from flask import Flask, render_template, request
import re
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lem = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
StopWords = stopwords.words("english")
import pandas as pd
import pickle
import numpy as np


# Load the Random Forest CLassifier model
# filename = 'CSV_suicide_intention.pkl'
# classify = pickle.load(open('NB_suicide_intention.pkl', 'rb'))
regressor = pickle.load(open('CSV_suicide_intention.pkl', 'rb'))
transform = pickle.load(open('TFID_transform.pkl', 'rb'))
# model = pickle.load(open('NB_suicide_intention.pkl.pkl', 'rb'))

# importing random module


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # temp_array = list()

    if request.method == 'POST':
        text_data = request.form['data']
        print(text_data)

        def get_clean(x):
            final = []
            text = re.sub('[^a-zA-Z]', " ", x)
            text = word_tokenize(text)
            lemm_text = [lem.lemmatize(w) for w in text if not w in StopWords]
            lemm_text = " ".join(lemm_text)
            final.append(lemm_text)
            return final

        info = get_clean(text_data)
        print(info)
        trans = transform.transform(info).toarray()
        print(trans)
        my_prediction = regressor.predict(trans)
        print(my_prediction)
        if my_prediction == [1]:
            # print('prediction',my_prediction)
            return render_template('new_suicide.html')
        else:
            return render_template('new_normal.html')


if __name__ == '__main__':
    app.run(debug=True)
