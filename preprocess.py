import pandas as pd
import re

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_missing_values(data):
    data = data.dropna(subset=['TITLE', 'CATEGORY'])
    return data

def normalize_text(data):
    def normalize(text):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        text = text.strip()
        return text
    data['TITLE'] = data['TITLE'].apply(normalize)
    return data

def extract_url_features(data):
    def extract_keywords(url):
        url = re.sub(r'https?://', '', url)
        url = re.sub(r'www\.', '', url)
        url = re.sub(r'\.[a-z]{2,3}/.*', '', url)
        keywords = re.split(r'\W+', url)
        return " ".join(keywords)
    
    data['URL_KEYWORDS'] = data['URL'].apply(extract_keywords)
    return data