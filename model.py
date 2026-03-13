import pickle
import numpy as np
from train_model import preprocess_text

class FakeNewsDetector:
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False

    def predict(self, text):
        if not self.is_loaded:
            return {"error": "Model not loaded properly."}
            
        clean_text = preprocess_text(text)
        if not clean_text:
            return {"error": "Text is empty after preprocessing."}

        # Vectorize
        vec_text = self.vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(vec_text)[0] # 1 for Fake, 0 for Real
        probabilities = self.model.predict_proba(vec_text)[0]
        confidence = probabilities[prediction] * 100
        
        # Explain reasoning (finding key words influencing the decision)
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Get active features in this text
        nonzero_elements = vec_text.nonzero()[1]
        
        word_scores = []
        for index in nonzero_elements:
            word = feature_names[index]
            score = coefficients[index] * vec_text[0, index]
            word_scores.append((word, score))
            
        # If Fake is 1 and Real is 0:
        # Positive score pushes toward Fake
        # Negative score pushes toward Real
        if prediction == 1:
            # Fake news, sort by highly positive scores
            word_scores.sort(key=lambda x: x[1], reverse=True)
            explanation_text = "The model detected sensational keywords and language patterns commonly associated with misinformation."
        else:
            # Real news, sort by highly negative scores
            word_scores.sort(key=lambda x: x[1])
            explanation_text = "The model detected formal language and credible patterns commonly associated with legitimate news."

        key_words = [word for word, score in word_scores[:5]] # Top 5 keywords

        label = "Fake News" if prediction == 1 else "Real News"

        return {
            "prediction": label,
            "confidence_score": round(confidence, 1),
            "key_words": key_words,
            "explanation": explanation_text
        }
