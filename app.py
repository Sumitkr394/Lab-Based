from flask import Flask, request, jsonify, render_template
import os
from model import FakeNewsDetector

app = Flask(__name__)

# Try to initialize the detector
detector = FakeNewsDetector('model.pkl', 'vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global detector
    if not detector.is_loaded:
        detector = FakeNewsDetector('model.pkl', 'vectorizer.pkl')
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    text = data['text']
    result = detector.predict(text)
    
    # Check if the model failed to load or predict
    if 'error' in result:
        return jsonify(result), 500
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
