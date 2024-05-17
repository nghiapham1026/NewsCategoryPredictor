import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import preprocess_cnn
import plot_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def prepare_data(X, max_length=512):
    # Tokenize and pad text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_length, padding='post')
    return X_pad, tokenizer.word_index

def build_model(input_length):
    # Create a Sequential model
    model = Sequential()
    
    # Add an embedding layer
    model.add(Embedding(
        input_dim=30000,  # Vocabulary size
        output_dim=128,   # Embedding vector size
        input_length=input_length  # Input sequence length
    ))
    
    # Add a Conv1D layer
    model.add(Conv1D(
        filters=64,
        kernel_size=5,
        activation='relu'
    ))
    
    # Add a GlobalMaxPooling1D layer
    model.add(GlobalMaxPooling1D())
    
    # Add a dense layer
    model.add(Dense(
        units=128,       
        activation='relu'
    ))
    
    # Add a dropout layer
    model.add(Dropout(
        rate=0.4          
    ))
    
    # Add an output layer with softmax activation for classification
    model.add(Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.00023864), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

from sklearn.metrics import confusion_matrix

def train_model(X, y, input_length):
    # Convert labels to one-hot encoding
    classes = np.unique(y)
    y_binarized = label_binarize(y, classes=classes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_model(input_length)
    
    # Define early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        validation_split=0.1,
        batch_size=8, 
        epochs=10, 
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model on the test set
    print("\nModel Performance on Test Set:")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Generate predictions
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
    y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels back to class labels

    # Compute the confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot ROC and Precision-Recall curves
    plot_metrics.plot_roc_curves("CNN Model", classes, y_test, y_pred_proba)
    plot_metrics.plot_precision_recall_curves("CNN Model", classes, y_test, y_pred_proba)

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess_cnn.load_data(file_path)  # Load the dataset
    data = preprocess_cnn.clean_missing_values(data)  # Clean missing values
    data['TITLE'] = data['TITLE'].apply(preprocess_cnn.normalize_text)  # Normalize text data
    
    X, word_index = prepare_data(data['TITLE'])  # Prepare input data
    y = data['CATEGORY'].astype('category').cat.codes  # Encode labels
    
    input_length = X.shape[1]  # Use the actual input length from prepared data
    
    train_model(X, y, input_length)  # Train the model

if __name__ == '__main__':
    main()

'''
Confusion Matrix:
[[180  12  15  15]
 [  9 270  17  11]
 [ 16   7  61   6]
 [ 28  10  11 177]]
'''