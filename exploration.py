import pandas as pd

def load_and_inspect_data(file_path):
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print("Data Information:")
    data.info()
    
    # Display the first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(data.head())

    return data

def perform_statistical_summaries(data):
    # Distribution of categories
    category_distribution = data['CATEGORY'].value_counts()
    print("\nCategory Distribution:")
    print(category_distribution)
    
    # Diversity of publishers
    publisher_diversity = data['PUBLISHER'].value_counts().head(10)
    print("\nTop 10 Publishers:")
    print(publisher_diversity)
    
    # Frequency of story groupings
    story_frequency = data['STORY'].value_counts().head(10)
    print("\nTop 10 Story Groupings:")
    print(story_frequency)

def main():
    file_path = './uci-news-aggregator.csv'
    data = load_and_inspect_data(file_path)
    perform_statistical_summaries(data)

if __name__ == "__main__":
    main()