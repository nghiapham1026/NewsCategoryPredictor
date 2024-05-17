import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(file_path):
    # Load dataset from a CSV file
    return pd.read_csv(file_path)

def clean_missing_values(data):
    # Drop rows with missing values in 'TITLE' or 'CATEGORY' columns
    data = data.dropna(subset=['TITLE', 'CATEGORY'])
    return data

def normalize_text(text):
    # Convert text to string and normalize it
    text = str(text).lower()  # Ensure text is converted to string and lowercase it
    text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with spaces
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def lemmatize_text(text):
    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize each token
    return ' '.join(lemmatized_tokens)  # Join tokens back into a single string

def extract_url_features(data):
    # Function to extract keywords from URL
    def extract_keywords(url):
        url = re.sub(r'https?://', '', url)  # Remove http or https protocol
        url = re.sub(r'www\.', '', url)  # Remove www
        url = re.sub(r'\.[a-z]{2,3}/.*', '', url)  # Remove domain suffix and everything after it
        keywords = re.split(r'\W+', url)  # Split URL into keywords
        return " ".join(keywords)
    
    # Apply keyword extraction to the 'URL' column
    data['URL_KEYWORDS'] = data['URL'].apply(extract_keywords)
    return data

def preprocess_data(input_file_path, output_file_path):
    # Load and preprocess data
    data = load_data(input_file_path)
    data = clean_missing_values(data)
    data['TITLE'] = data['TITLE'].apply(normalize_text)  # Normalize text in 'TITLE' column
    
    # Define custom stopwords and combine with NLTK stopwords
    custom_stopwords = set(['new', 'u', 'to', 'the', 'in', 'and', 'of', 'a', 'for', 'on', 'with', 'at', 'is', 'that', 'it', 'this'])
    all_stopwords = set(stopwords.words('english')) | custom_stopwords

    # Function to preprocess the title
    def preprocess_title(title):
        words = nltk.word_tokenize(title)  # Tokenize the title
        filtered_words = [word for word in words if word.lower() not in all_stopwords]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]  # Lemmatize words
        return ' '.join(lemmatized_words)  # Join words back into a single string
    
    # Apply title preprocessing to the 'TITLE' column
    data['TITLE'] = data['TITLE'].apply(preprocess_title)
    
    # Extract URL features
    data = extract_url_features(data)
    
    # Save preprocessed data to a new CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Preprocessing completed. Preprocessed data saved to {output_file_path}")

if __name__ == '__main__':
    # File paths for input and output data
    input_file_path = './uci-news-aggregator_small.csv'
    output_file_path = './preprocessed_data.csv'
    
    # Preprocess the data
    preprocess_data(input_file_path, output_file_path)