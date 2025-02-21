from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download all required NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")


# Download NLTK data at startup
download_nltk_data()

app = Flask(__name__)
CORS(app)


# Text preprocessing function
def preprocess_text(text):
    try:
        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Join tokens back into text
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing text: {str(e)}")
        return text  # Return original text if preprocessing fails


# Load and train the model
def train_model():
    try:
        print("Loading training data...")
        # Load your training data
        df = pd.read_csv('train.csv')

        # Clean the data
        df = df.dropna()
        print(f"Loaded {len(df)} rows of training data")

        print("Preprocessing text...")
        # Preprocess the text
        df['cleaned_text'] = df['text'].apply(preprocess_text)

        print("Creating TF-IDF features...")
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['label']  # Assuming 'label' is your target column

        print("Training logistic regression model...")
        # Train the model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        print("Saving model and vectorizer...")
        # Save the model and vectorizer
        joblib.dump(model, 'fake_news_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

        return model, vectorizer
    except Exception as e:
        print(f"Error in training model: {str(e)}")
        raise


# Load or train the model
try:
    print("Attempting to load existing model...")
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Successfully loaded existing model and vectorizer")
except FileNotFoundError:
    print("No existing model found. Training new model...")
    model, vectorizer = train_model()
    print("Model training completed")
except Exception as e:
    print(f"Error loading/training model: {str(e)}")
    raise


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the text
        cleaned_text = preprocess_text(text)

        # Transform text using saved vectorizer
        text_features = vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(text_features)[0]
        probability = model.predict_proba(text_features)[0]

        # Return prediction and confidence
        response = {
            'prediction': bool(prediction),
            'confidence': float(max(probability)),
            'message': 'Fake news detected' if prediction else 'Genuine news'
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)