import pandas as pd
import string
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Removing stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv("news.csv")
    except Exception as e:
        print(f"Error loading news.csv: {e}")
        return

    # --- INJECT SYNTHETIC DATA ---
    print("Injecting synthetic data...")
    synthetic_data = [
        # Universal Truths / Short facts (Real = 0)
        {"text": "The earth is round.", "label": "Real"},
        {"text": "Water boils at 100 degrees Celsius.", "label": "Real"},
        {"text": "The sun rises in the east.", "label": "Real"},
        {"text": "Humans need oxygen to survive.", "label": "Real"},
        {"text": "Paris is the capital of France.", "label": "Real"},
        {"text": "A triangle has three sides.", "label": "Real"},
        {"text": "Dogs are mammals.", "label": "Real"},
        {"text": "Gravity pulls objects toward the center of the Earth.", "label": "Real"},
        {"text": "The moon orbits the Earth.", "label": "Real"},
        {"text": "Ice is frozen water.", "label": "Real"},
        {"text": "It is what it is.", "label": "Real"},
        {"text": "This is a fact.", "label": "Real"},
        {"text": "I am a human.", "label": "Real"},
        {"text": "Fire is hot.", "label": "Real"},
        {"text": "The sky is blue.", "label": "Real"},
        
        # Fake / Conspiracy theories (Fake = 1)
        {"text": "The earth is flat.", "label": "Fake"},
        {"text": "Lizard people rule the government secretly.", "label": "Fake"},
        {"text": "Vaccines cause magnetism in human bodies.", "label": "Fake"},
        {"text": "The moon landing was faked on a soundstage.", "label": "Fake"},
        {"text": "Birds aren't real, they are government drones.", "label": "Fake"},
        {"text": "5G towers cause viral infections.", "label": "Fake"},
        {"text": "The sky is green.", "label": "Fake"},
        {"text": "Drinking bleach cures all diseases.", "label": "Fake"},
        {"text": "Aliens built the pyramids.", "label": "Fake"},
        {"text": "Elvis is still alive.", "label": "Fake"},
        {"text": "The government controls the weather with a machine.", "label": "Fake"},
        {"text": "Dinosaurs never existed.", "label": "Fake"},
        {"text": "The sun is cold.", "label": "Fake"},
        {"text": "Gravity is an illusion.", "label": "Fake"}
    ]
    # Multiply by 50 to give the synthetic data enough weight to influence the model
    df_synthetic = pd.DataFrame(synthetic_data * 50)
    df = pd.concat([df, df_synthetic], ignore_index=True)
    # -----------------------------

    print("Preprocessing data...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Encode labels: Fake = 1, Real = 0
    df['label_num'] = df['label'].map({'Fake': 1, 'Real': 0})
    
    # Check if there are unmapped labels
    if df['label_num'].isnull().any():
        print("Warning: Some labels could not be mapped. Make sure labels are 'Fake' or 'Real'.")
        df = df.dropna(subset=['label_num'])

    X = df['clean_text']
    y = df['label_num']

    if len(df) == 0:
        print("Dataset is empty or incorrectly formatted.")
        return

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], labels=[0, 1], zero_division=0)
    print(report)

    print("\nSaving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Training complete! model.pkl and vectorizer.pkl saved.")

if __name__ == "__main__":
    main()
