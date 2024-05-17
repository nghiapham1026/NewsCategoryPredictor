import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download NLTK resources (run once)
nltk.download('punkt')

def load_and_inspect_data(file_path):
    # Load dataset from a CSV file
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print("Data Information:")
    print(data.info())
    
    # Display the first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(data.head())

    return data

def perform_statistical_summaries(data):
    # Ensure 'TITLE' column contains only string values
    data['TITLE'] = data['TITLE'].astype(str)  # Convert any non-string values to strings
    
    # Distribution of categories
    category_distribution = data['CATEGORY'].value_counts()
    print("\nCategory Distribution:")
    print(category_distribution)
    
    # Plot category distribution
    plt.figure(figsize=(8, 6))
    category_distribution.plot(kind='bar', color='skyblue')
    plt.title('Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # Tokenize titles and calculate word frequency
    title_words = data['TITLE'].apply(word_tokenize)  # Tokenize each title
    all_words = [word for sublist in title_words for word in sublist]  # Flatten the list of lists
    word_counts = Counter(all_words)  # Count frequency of each word
    
    # Get top 3 most frequent words
    top_words = word_counts.most_common(3)  # Get top 3 most common words
    print("\nTop 3 Most Frequently Seen Words in Titles:")
    for word, count in top_words:
        print(f"{word}: {count} times")

def main():
    input_file_path = './preprocessed_data.csv'
    
    # Load preprocessed data
    data = load_and_inspect_data(input_file_path)
    
    # Perform statistical summaries on preprocessed data
    perform_statistical_summaries(data)

if __name__ == "__main__":
    main()