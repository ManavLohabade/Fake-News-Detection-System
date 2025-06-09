import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/finalized_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'\W', ' ', text) # Remove all non-word characters
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news input from the request
    news_text = request.form['news_text']

    # Handle empty input
    if not news_text.strip():
        return jsonify({'prediction': 'Please enter news to predict.'})

    # Preprocess the news text
    cleaned_news_text = preprocess_text(news_text)

    # Transform the input text using the loaded TF-IDF vectorizer
    vectorized_news = vectorizer.transform([cleaned_news_text])

    # Make the prediction using the model
    prediction = model.predict(vectorized_news)

    print(f"Prediction value: {prediction[0]}")
    print(f"Prediction type: {type(prediction[0])}")

    # Return the result as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
