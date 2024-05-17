import pandas as pd
import re

def load_data(file_path):
    # Load dataset from a CSV file
    return pd.read_csv(file_path)

def clean_missing_values(data):
    # Drop rows with missing values in 'TITLE' or 'CATEGORY' columns
    data = data.dropna(subset=['TITLE', 'CATEGORY'])
    return data

def normalize_text(data):
    # Function to normalize text
    def normalize(text):
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with spaces
        text = text.strip()  # Remove leading and trailing whitespace
        return text
    
    # Apply normalization to the 'TITLE' column
    data['TITLE'] = data['TITLE'].apply(normalize)
    return data

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