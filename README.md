# News Article Classification Project

This project involves building and comparing machine learning models to classify news articles into categories such as business, technology, entertainment, and health. The project includes scripts for data exploration, preprocessing, baseline model training, hyperparameter tuning, and advanced model training using a CNN and BERT.

## Project Structure

- `exploration.py`: Script for exploring and analyzing the dataset.
- `ml_baseline.py`: Script for training baseline machine learning models and tuning them using grid search.
- `ml_cnn.py`: Script for building and training a CNN model integrated with BERT for advanced text classification **(VERY INTENSIVE)**.
- `preprocess.py`: Utility script for data preprocessing including text normalization and feature extraction.
- `subset.py`: Utility script to create a smaller subset of the dataset for quicker training.
- `plot_metrics.py`: Functions to plot ROC and Precision-Recall curves for a classification model.
- `cnn_plot.py`: Plots to visualize training and validation metrics (accuracy and loss) across epochs for model.

## Setup and Usage

### Dependencies

- Python 3.8+
- Pandas
- Scikit-learn
- TensorFlow
- Transformers (Hugging Face)

To install the necessary libraries, run:
```bash
pip install pandas scikit-learn tensorflow transformers
```

### Running the Scripts

1. **Preprocessing**:
   Simply import the `preprocess.py` module in other scripts as needed for data cleaning and feature extraction tasks. Import the `plot_metrics.py` module to plot ROC and Precision-Recall Curve for your model.

2. **Creating a Data Subset**:
   ```bash
   python subset.py
   ```
   Use this script to generate a smaller subset of the dataset for development and testing.

3. **Data Exploration**:
   ```bash
   python exploration.py
   ```
   This script will load the data, display initial statistics, and perform category and publisher analyses.

4. **Baseline Models**:
   ```bash
   python ml_baseline.py
   ```
   Trains Logistic Regression, Decision Tree, and SVM models. It includes hyperparameter tuning with grid search.

5. **CNN Model Standalone**:
   ```bash
   python ml_cnn.py
   ```
   Trains a deep learning model using Convolutional Neural Networks without BERT.

6. **CNN with BERT Model**:
   ```bash
   python ml_bert.py
   ```
   Trains a deep learning model using Convolutional Neural Networks and BERT for text classification. This model is very taxing on your CPU!

## Data

Data used in this project is the UCI News Aggregator dataset which consists of summaries of news articles, categorized into different types like business, technology, and more.
