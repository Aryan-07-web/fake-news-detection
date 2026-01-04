# Fake vs Real News Detection using NLP and Machine Learning

A supervised machine learning system that classifies news articles as REAL or FAKE using NLP techniques and a trained ML model.

## Features

- **NLP Preprocessing**: Lowercasing, tokenization, stopword removal, and lemmatization
- **Feature Extraction**: TF-IDF with unigrams and bigrams
- **Machine Learning**: Logistic Regression classifier (can be switched to Naive Bayes)
- **Model Evaluation**: Accuracy, Precision, Recall, and F1-score metrics
- **Prediction**: Real-time text classification with confidence scores
- **CLI Interface**: Interactive command-line interface for predictions
- **Streamlit UI**: Modern web interface for easy text classification

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. The script will automatically download required NLTK data on first run.

## Dataset

The script will automatically create a sample dataset if `dataset.csv` doesn't exist. For better performance, use a larger publicly available dataset:

### Recommended Public Datasets

1. **Fake and Real News Dataset (Kaggle)** - Most Popular
   - Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - Contains `True.csv` and `Fake.csv` files
   - Combine them into `dataset.csv` with 'text' and 'label' columns

2. **ISOT Fake News Dataset**
   - Available on Kaggle and GitHub
   - Well-structured with labeled news articles

3. **LIAR Dataset**
   - For political fake news detection
   - Available on GitHub

### Dataset Format

The dataset should be a CSV file named `dataset.csv` with the following format:

```csv
text,label
"Scientists have discovered a new planet...",real
"BREAKING: Aliens have made contact...",fake
```

**Column names**: The script automatically detects columns containing "text", "article", "news", "content" (for text) and "label", "class", "category" (for labels).

**Combining Kaggle Dataset** (if using Fake and Real News Dataset):
```python
import pandas as pd

# Load the two files
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

# Add labels
true_news['label'] = 'real'
fake_news['label'] = 'fake'

# Combine (using 'text' column or appropriate text column)
# Adjust column name as needed (might be 'text', 'title', 'text', etc.)
combined = pd.concat([
    true_news[['text', 'label']],  # Replace 'text' with actual column name
    fake_news[['text', 'label']]
], ignore_index=True)

# Shuffle
combined = combined.sample(frac=1).reset_index(drop=True)

# Save
combined.to_csv('dataset.csv', index=False)
```

**Label formats supported**:
- Real/Fake
- Truthful/Deceptive
- True/False
- 1/0 (numeric)
- Any binary classification format

## Usage

### Basic Usage

#### CLI Mode (Command Line)

Run the script to train the model and enter interactive prediction mode:

```bash
python fake_real_classifier.py --cli
```

Or simply:
```bash
python fake_real_classifier.py
```

The script will:
1. Load the dataset from `dataset.csv`
2. Preprocess the text data
3. Train the ML model
4. Evaluate the model on test data
5. Enter interactive mode for predictions

#### Streamlit Web UI

For a modern web interface, use Streamlit:

```bash
streamlit run fake_real_classifier.py
```

Or install Streamlit first (if not already installed):
```bash
pip install streamlit
streamlit run fake_real_classifier.py
```

The Streamlit interface provides a simple, user-friendly web interface with:
- One-click model training
- Text input for classification
- Real-time predictions with confidence scores

### Programmatic Usage

```python
from fake_real_classifier import FakeRealClassifier, load_dataset

# Load dataset
texts, labels = load_dataset('dataset.csv')

# Initialize classifier
classifier = FakeRealClassifier(model_type='logistic')  # or 'naive_bayes'

# Train the model
classifier.train(texts[:800], labels[:800])

# Evaluate
classifier.evaluate(texts[800:], labels[800:])

# Make predictions
label, confidence = classifier.predict("Scientists discover new planet in nearby star system.", return_probability=True)
print(f"Prediction: {label}, Confidence: {confidence:.2%}")
```

## Model Configuration

You can change the model type by modifying the `MODEL_TYPE` variable in the `main()` function:

- `'logistic'`: Logistic Regression (default)
- `'naive_bayes'`: Multinomial Naive Bayes

## Project Structure

```
.
├── fake_real_classifier.py  # Main script with all functionality
├── download_dataset.py      # Helper script to download and prepare public dataset
├── dataset.csv              # Combined dataset (created by download_dataset.py)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Technical Details

### NLP Pipeline

1. **Lowercasing**: Convert all text to lowercase
2. **Tokenization**: Split text into individual words
3. **Stopword Removal**: Remove common words (the, a, an, etc.)
4. **Lemmatization**: Convert words to their base forms (running → run)

### Feature Extraction

- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **N-grams**: Unigrams (1 word) and bigrams (2 words)
- **Max Features**: 5000 most important features

### Model Training

- **Algorithm**: Logistic Regression (or Naive Bayes)
- **Train/Test Split**: 80/20 by default
- **Stratified Split**: Maintains class distribution

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Code Statistics

- **Total Lines**: ~400 lines
- **Single File**: All functionality in one Python file
- **Clear Comments**: Detailed explanations for each step

## Limitations

- No deep learning models
- No transformer models
- Requires labeled training data
- Performance depends on dataset quality and size

## License

This project is provided as-is for educational purposes.

