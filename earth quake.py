import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the fake news dataset from a CSV file
dataset = pd.read_csv('fake_news_dataset.csv')

# Extract text and labels from the dataset
texts = dataset['text']
labels = dataset['label']

# Tokenization and Text Preprocessing
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization and lowercase conversion
    words = word_tokenize(text.lower())
    
    # Remove non-alphanumeric characters and stopwords
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    return ' '.join(words)

# Apply the preprocessing function to all text data
texts = texts.apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF Vectorization (Feature Extraction)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
