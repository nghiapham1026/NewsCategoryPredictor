import pandas as pd

def subset_csv(input_file, output_file, fraction=0.1):
    df = pd.read_csv(input_file)

    # Randomly sample the fraction of rows
    subset_df = df.sample(frac=fraction, random_state=1)  # random_state ensures reproducibility

    # Save the subset to a new CSV file
    subset_df.to_csv(output_file, index=False)

    print(f"Subset created successfully and saved to {output_file}")

if __name__ == "__main__":
    input_csv_path = './uci-news-aggregator.csv'
    output_csv_path = './uci-news-aggregator_small.csv'

    subset_csv(input_csv_path, output_csv_path)