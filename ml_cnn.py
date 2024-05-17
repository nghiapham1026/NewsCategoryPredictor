import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import preprocess_cnn
from sklearn.model_selection import train_test_split

def prepare_data(X, max_length=512):
    # Tokenize and pad text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_length, padding='post')
    return X_pad, tokenizer.word_index

def build_model(hp, input_length):
    # Define a Sequential model
    model = Sequential()
    
    # Add an Embedding layer with hyperparameters
    model.add(Embedding(
        input_dim=hp.Int('input_dim', min_value=10000, max_value=50000, step=10000), 
        output_dim=hp.Choice('output_dim', values=[64, 128, 256]),
        input_length=input_length
    ))
    
    # Add a Conv1D layer with hyperparameters
    model.add(Conv1D(
        filters=hp.Choice('filters', values=[64, 128, 256]),
        kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]),
        activation='relu'
    ))
    
    # Add a GlobalMaxPooling1D layer
    model.add(GlobalMaxPooling1D())
    
    # Add a Dense layer with hyperparameters
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64), 
        activation='relu'
    ))
    
    # Add a Dropout layer with hyperparameters
    model.add(Dropout(
        rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    ))
    
    # Add an output layer with softmax activation for classification
    model.add(Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_model(X, y, input_length):
    # Define a RandomSearch tuner
    tuner = RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_length),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='model_tuning',
        project_name='NewsCategoryTuning'
    )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Search for the best hyperparameters
    tuner.search(X_train, y_train, epochs=10, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
    
    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Evaluate the best model on the test set
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess_cnn.load_data(file_path)  # Load the dataset
    data = preprocess_cnn.clean_missing_values(data)  # Clean missing values
    data['TITLE'] = data['TITLE'].apply(preprocess_cnn.normalize_text)  # Normalize text data
    
    # Prepare input data
    X, word_index = prepare_data(data['TITLE'])
    y = data['CATEGORY'].astype('category').cat.codes  # Encode labels
    
    # Use the actual input length from prepared data
    input_length = X.shape[1]
    
    # Tune and evaluate the model
    tune_model(X, y, input_length)

if __name__ == '__main__':
    main()