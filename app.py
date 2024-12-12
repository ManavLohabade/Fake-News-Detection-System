import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/finalized_model.pkl', 'rb'))

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

    # Reshape the input text to 2D (1 sample with 1 feature)
    input_data = np.array([news_text]).reshape(1, -1)  # Reshaping to 2D array

    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Return the result as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
