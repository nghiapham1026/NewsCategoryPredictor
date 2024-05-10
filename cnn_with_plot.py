import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import preprocess
import plot_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def prepare_data(X, max_length=512):
    # Tokenize text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_length, padding='post')
    return X_pad, tokenizer.word_index

def build_model(input_length):
    model = Sequential()
    model.add(Embedding(
        input_dim=30000, 
        output_dim=128,  
        input_length=input_length
    ))
    model.add(Conv1D(
        filters=64,
        kernel_size=5,
        activation='relu'
    ))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(
        units=128,       
        activation='relu'
    ))
    model.add(Dropout(
        rate=0.4          
    ))
    model.add(Dense(4, activation='softmax'))  # Ensure the output layer is suitable for classification
    
    model.compile(
        optimizer=Adam(learning_rate=0.00023864), 
        loss='categorical_crossentropy',  # Change here
        metrics=['accuracy']
    )
    return model

def train_model(X, y, input_length):
    # Convert labels to categorical as needed by sklearn's label_binarize
    classes = np.unique(y)
    y_binarized = label_binarize(y, classes=classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)
    
    model = build_model(input_length)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        validation_split=0.1,
        batch_size=8, 
        epochs=10, 
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model to get the loss and accuracy
    print("\nModel Performance on Test Set:")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Generate predictions for plotting
    y_pred_proba = model.predict(X_test)
    plot_metrics.plot_roc_curves("CNN Model", classes, y_test, y_pred_proba)
    plot_metrics.plot_precision_recall_curves("CNN Model", classes, y_test, y_pred_proba)

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data['TITLE'] = data['TITLE'].apply(preprocess.normalize_text)
    
    X, word_index = prepare_data(data['TITLE'])
    y = data['CATEGORY'].astype('category').cat.codes
    
    input_length = X.shape[1]  # Use the actual input length from prepared data
    
    train_model(X, y, input_length)

if __name__ == '__main__':
    main()