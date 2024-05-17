# News Aggregator Machine Learning Project

This project focuses on building and evaluating machine learning models to classify news articles into categories. The project includes preprocessing steps, exploratory data analysis, baseline machine learning models, a Convolutional Neural Network (CNN) model, and performance visualization tools.

## Project Structure

- **preprocess.py**: Contains functions for loading data, cleaning missing values, normalizing text, and extracting URL features.
- **preprocess_cnn.py**: Extends preprocessing steps with text tokenization, lemmatization, and stopword removal specifically for CNN model training.
- **exploration.py**: Provides functions to load, inspect, and perform statistical summaries on the dataset, including visualizing category distribution and identifying the most frequent words in titles.
- **ml_baseline.py**: Implements and tunes baseline machine learning models (Logistic Regression, Decision Tree, and Support Vector Machine) using TF-IDF vectorization.
- **ml_cnn.py**: Defines and tunes a Convolutional Neural Network model, including hyperparameter optimization and performance evaluation.
- **cnn_with_plot.py**: Defines and trains a Convolutional Neural Network model, including data preparation and performance evaluation.
- **plot_metrics.py**: Provides functions for plotting ROC curves and Precision-Recall curves for model performance evaluation.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- keras-tuner
- nltk
- matplotlib

## Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/news-aggregator-ml.git
   cd news-aggregator-ml
   ```

2. **Install the required packages**:
   ```sh
   pip install pandas numpy scikit-learn tensorflow keras-tuner nltk matplotlib
   ```

3. **Download NLTK resources**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

## Usage

### Data Preprocessing

**Preprocess data for baseline models**:
```sh
python preprocess.py
```

**Preprocess data for CNN model**:
```sh
python preprocess_cnn.py
```

### Exploratory Data Analysis

**Explore and summarize the dataset**:
```sh
python exploration.py
```

### Training and Evaluation

**Train and evaluate baseline models**:
```sh
python ml_baseline.py
```

**Train and evaluate CNN model with hyperparameter tuning**:
```sh
python ml_cnn.py
```

**Train and evaluate CNN model with predefined parameters**:
```sh
python cnn_with_plot.py
```

### Visualizing Model Performance

The `plot_metrics.py` script contains functions to plot ROC curves and Precision-Recall curves to visualize the performance of the trained models. These functions are called automatically during model training and evaluation in `ml_baseline.py` and `cnn_with_plot.py`.

## Functions Overview

### preprocess.py

- `load_data(file_path)`: Loads data from a CSV file.
- `clean_missing_values(data)`: Removes rows with missing values.
- `normalize_text(data)`: Converts text to lowercase, removes non-word characters, and strips whitespace.
- `extract_url_features(data)`: Extracts keywords from URLs.

### preprocess_cnn.py

- `load_data(file_path)`: Loads data from a CSV file.
- `clean_missing_values(data)`: Removes rows with missing values.
- `normalize_text(text)`: Converts text to lowercase, removes non-word characters, and strips whitespace.
- `lemmatize_text(text)`: Lemmatizes tokens in the text.
- `extract_url_features(data)`: Extracts keywords from URLs.
- `preprocess_data(input_file_path, output_file_path)`: Applies all preprocessing steps and saves the cleaned data.

### exploration.py

- `load_and_inspect_data(file_path)`: Loads and inspects the dataset, displaying basic information and the first few rows.
- `perform_statistical_summaries(data)`: Displays and plots the category distribution, and identifies the top 3 most frequent words in titles.

### ml_baseline.py

- `train_and_tune_models(X, y)`: Trains and tunes Logistic Regression, Decision Tree, and SVM models using GridSearchCV.
- `main()`: Loads data, preprocesses it, and trains models.

### ml_cnn.py

- `prepare_data(X, max_length=512)`: Tokenizes and pads text data.
- `build_model(hp, input_length)`: Builds the CNN model with hyperparameters.
- `tune_model(X, y, input_length)`: Tunes the CNN model using RandomSearch and evaluates its performance.
- `main()`: Loads data, preprocesses it, and tunes the CNN model.

### cnn_with_plot.py

- `prepare_data(X, max_length=512)`: Tokenizes and pads text data.
- `build_model(input_length)`: Builds the CNN model.
- `train_model(X, y, input_length)`: Trains the CNN model and evaluates its performance.
- `main()`: Loads data, preprocesses it, and trains the CNN model.

### plot_metrics.py

- `plot_roc_curves(model_name, classes, y_test_binarized, y_pred_proba)`: Plots ROC curves for each class.
- `plot_precision_recall_curves(model_name, classes, y_test_binarized, y_pred_proba)`: Plots Precision-Recall curves for each class.

## Contact

For questions or feedback, please contact Nathan Pham or Shreya Raj.