import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# For downloading public datasets
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Streamlit import (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class FakeRealClassifier:
    """
    A classifier for distinguishing between fake and real news articles.
    
    This class handles:
    - NLP preprocessing (lowercasing, tokenization, stopword removal, lemmatization)
    - TF-IDF feature extraction (unigrams and bigrams)
    - Model training (Logistic Regression)
    - Model evaluation
    - News article prediction with confidence scores
    """
    
    def __init__(self, model_type='logistic'):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of model to use ('logistic' or 'naive_bayes')
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.model = None
        self.model_type = model_type
        self.is_trained = False
        
    def preprocess_text(self, text):
        """
        Perform NLP preprocessing on input text.
        
        Steps:
        1. Lowercasing
        2. Tokenization
        3. Stopword removal
        4. Lemmatization
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Step 1: Lowercasing
        text = text.lower()
        
        # Step 2: Tokenization
        tokens = word_tokenize(text)
        
        # Step 3: Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        # Step 4: Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def preprocess_dataset(self, texts):
        """
        Preprocess a list of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of preprocessed text strings
        """
        preprocessed_texts = []
        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            preprocessed_texts.append(preprocessed_text)
        return preprocessed_texts
    
    def train(self, X_train, y_train):
        """
        Train the machine learning model on the dataset.
        
        This function:
        1. Preprocesses the training texts
        2. Converts text to TF-IDF features (unigrams and bigrams)
        3. Trains the selected ML classifier
        
        Args:
            X_train (list/pd.Series): Training text data
            y_train (list/pd.Series): Training labels (real/fake)
        """
        print("Starting NLP preprocessing...")
        # Preprocess training texts
        X_train_preprocessed = self.preprocess_dataset(X_train)
        
        print("Converting text to TF-IDF features (unigrams and bigrams)...")
        # Convert text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train_preprocessed)
        
        print(f"Training {self.model_type} classifier...")
        # Initialize and train the model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError("model_type must be 'logistic' or 'naive_bayes'")
        
        # Train the model
        self.model.fit(X_train_tfidf, y_train)
        self.is_trained = True
        print("Model training completed!")
    
    def predict(self, text, return_probability=False):
        """
        Predict if a text is REAL or FAKE using the trained model.
        
        Args:
            text (str): Input text to classify
            return_probability (bool): If True, return confidence score
            
        Returns:
            str or tuple: Predicted label ('Real' or 'Fake'), optionally with confidence score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Preprocess the input text
        preprocessed_text = self.preprocess_text(text)
        
        # Convert to TF-IDF features
        text_tfidf = self.vectorizer.transform([preprocessed_text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)[0]
        label = 'Real' if prediction == 1 else 'Fake'
        
        if return_probability:
            # Get prediction probabilities
            probabilities = self.model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            return label, confidence
        else:
            return label
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Computes:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Detailed classification report
        
        Args:
            X_test (list/pd.Series): Test text data
            y_test (list/pd.Series): Test labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        print("\nEvaluating model on test data...")
        # Preprocess test texts
        X_test_preprocessed = self.preprocess_dataset(X_test)
        
        # Convert to TF-IDF features
        X_test_tfidf = self.vectorizer.transform(X_test_preprocessed)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*50)
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, filepath):
        """Save the trained model and vectorizer to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and vectorizer from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def download_public_dataset(output_path='dataset.csv', max_samples=1000):
    """
    Download and prepare a public fake news dataset.
    Creates a combined dataset from publicly available sources.
    
    Note: This creates a sample dataset. For full datasets, download from:
    - Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    - Or use the sample dataset.csv provided
    
    Args:
        output_path (str): Path to save the combined dataset
        max_samples (int): Maximum samples per class (for quick testing)
    
    Returns:
        bool: True if dataset was created, False otherwise
    """
    print("Creating dataset from public sources...")
    print("Note: For larger datasets, download from Kaggle or other public repositories.")
    
    # Sample dataset structure - users can replace with downloaded datasets
    sample_data = {
        'text': [
            # Real news samples
            "Scientists have discovered a new planet in a nearby star system that could potentially support life.",
            "A major breakthrough in renewable energy has been achieved as engineers develop solar panels with improved efficiency.",
            "Economic data shows steady growth in employment rates across multiple sectors.",
            "Medical researchers have successfully tested a new vaccine showing promise in preventing cancer.",
            "Climate scientists report that global temperatures have increased requiring immediate action.",
            "International space agencies announce plans to build a permanent research base on the Moon.",
            "Educational institutions implement new programs to improve digital literacy among students.",
            "Technology companies invest heavily in artificial intelligence research for healthcare applications.",
            # Fake news samples
            "BREAKING: Aliens have made contact with Earth and are demanding surrender immediately.",
            "SHOCKING: Scientists discovered that drinking water causes cancer in 100 percent of people.",
            "URGENT: Government is putting tracking chips in vaccines to control everyone's minds.",
            "EXCLUSIVE: NASA admits the Moon landing was fake and filmed in Hollywood studio.",
            "ALERT: Eating bananas prevents coronavirus completely according to secret doctors.",
            "REVEALED: The Earth is flat and NASA has been lying for decades with CGI photos.",
            "WARNING: Your phone is spying 24/7 and tech companies sell conversations to government.",
            "EXPOSED: Celebrities use secret mind control techniques to manipulate the public.",
        ],
        'label': ['real'] * 8 + ['fake'] * 8
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created at {output_path}")
    print(f"Total samples: {len(df)} (Real: {len(df[df['label']=='real'])}, Fake: {len(df[df['label']=='fake'])})")
    print("\nTo use a larger public dataset:")
    print("1. Download from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("2. Combine True.csv and Fake.csv into dataset.csv with 'text' and 'label' columns")
    print("3. Or use any CSV file with text and label columns")
    
    return True


def load_dataset(csv_path='dataset.csv', auto_download=True):
    """
    Load the dataset from a CSV file. If file doesn't exist and auto_download is True,
    creates a sample dataset.
    
    Expected CSV format:
    - One column with news article text content
    - One column with labels (can be: real/fake, truthful/deceptive, 0/1, etc.)
    
    Args:
        csv_path (str): Path to the CSV file
        auto_download (bool): If True, create sample dataset if file doesn't exist
        
    Returns:
        tuple: (texts, labels) as pandas Series
    """
    # Check if dataset exists, if not, create sample
    if not os.path.exists(csv_path) and auto_download:
        print(f"Dataset file '{csv_path}' not found.")
        download_public_dataset(csv_path)
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Try to identify text and label columns automatically
    # Common column names
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['text', 'review', 'content', 'message', 'comment', 'article', 'news']):
            text_col = col
        if any(keyword in col_lower for keyword in ['label', 'class', 'category', 'type']):
            label_col = col
    
    # If not found, use first two columns
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1]
    
    texts = df[text_col]
    labels = df[label_col]
    
    # Convert labels to binary (0 for fake, 1 for real)
    # Handle different label formats
    if labels.dtype == 'object':
        labels_lower = labels.str.lower()
        labels_binary = labels_lower.map({
            'fake': 0, 'deceptive': 0, 'false': 0,
            'real': 1, 'truthful': 1, 'true': 1
        })
        # If mapping failed, try to infer from unique values
        if labels_binary.isna().any():
            unique_labels = labels_lower.unique()
            if len(unique_labels) == 2:
                labels_binary = (labels_lower == unique_labels[1]).astype(int)
    else:
        labels_binary = labels
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"Label distribution: {labels_binary.value_counts().to_dict()}")
    
    return texts, labels_binary


def streamlit_app():
    """
    Simple Streamlit web application interface for fake news detection.
    """
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    
    st.title("Fake vs Real News Detector")
    st.write("Detect fake news articles using NLP and Machine Learning")
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
        st.session_state.is_trained = False
    
    # Train model button
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model... Please wait."):
            try:
                import io
                import contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    texts, labels = load_dataset('dataset.csv')
                    X_train, X_test, y_train, y_test = train_test_split(
                        texts, labels, test_size=0.2, random_state=42, stratify=labels
                    )
                    classifier = FakeRealClassifier(model_type='logistic')
                    classifier.train(X_train, y_train)
                    metrics = classifier.evaluate(X_test, y_test)
                
                st.session_state.classifier = classifier
                st.session_state.is_trained = True
                st.success(f"Model trained! Accuracy: {metrics['accuracy']:.2%}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Text input for prediction
    user_text = st.text_area("Enter news article to classify:", height=150)
    
    if st.button("Classify", disabled=not st.session_state.is_trained):
        if not st.session_state.is_trained:
            st.warning("Please train the model first!")
        elif not user_text.strip():
            st.warning("Please enter some text!")
        else:
            try:
                label, confidence = st.session_state.classifier.predict(user_text, return_probability=True)
                if label == "Real":
                    st.success(f"**Prediction: {label}** (Confidence: {confidence:.2%})")
                else:
                    st.error(f"**Prediction: {label}** (Confidence: {confidence:.2%})")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


def main():
    """
    Main function to run the fake news detection pipeline (CLI mode).
    """
    print("="*60)
    print("Fake vs Real News Detection System")
    print("="*60)
    
    # Configuration
    DATASET_PATH = 'dataset.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODEL_TYPE = 'logistic'  # Options: 'logistic' or 'naive_bayes'
    
    # Load dataset
    try:
        texts, labels = load_dataset(DATASET_PATH)
    except FileNotFoundError:
        print(f"\nError: Dataset file '{DATASET_PATH}' not found!")
        print("Please ensure the CSV file exists with text and label columns.")
        return
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        return
    
    # Split dataset into training and testing sets
    print(f"\nSplitting dataset: {TEST_SIZE*100}% for testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize classifier
    classifier = FakeRealClassifier(model_type=MODEL_TYPE)
    
    # Train the model
    classifier.train(X_train, y_train)
    
    # Evaluate the model
    metrics = classifier.evaluate(X_test, y_test)
    
    # Interactive prediction mode (CLI)
    print("\n" + "="*60)
    print("Interactive Prediction Mode")
    print("="*60)
    print("Enter news article to classify as Real or Fake (type 'quit' to exit)")
    print("-"*60)
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        try:
            label, confidence = classifier.predict(user_input, return_probability=True)
            print(f"\nPrediction: {label}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    print("\nThank you for using the Fake News Detector!")


# Entry point
# For CLI: Run with 'python fake_real_classifier.py --cli'
# For Streamlit: Run with 'streamlit run fake_real_classifier.py'
if __name__ == "__main__":
    import sys
    if '--cli' in sys.argv or not STREAMLIT_AVAILABLE:
        main()
    else:
        streamlit_app()

