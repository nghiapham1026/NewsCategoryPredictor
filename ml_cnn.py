import numpy as np
import tensorflow as tf
import preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data(X, max_length=512):
    # Convert text data to sequences of integers
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_length, padding='post')
    return X_pad

def build_model(input_shape, num_classes, learning_rate=5e-5):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=input_shape),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model(X, y, model, batch_size=8, epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping], verbose=1)
    print("\nModel Performance on Test Set:")
    model.evaluate(X_test, y_test)

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data = preprocess.normalize_text(data)
    data = preprocess.extract_url_features(data)
    
    X = prepare_data(data['TITLE'])
    y = data['CATEGORY'].astype('category').cat.codes
    
    model = build_model(input_shape=X.shape[1], num_classes=len(np.unique(y)))
    train_model(X, y, model)

if __name__ == '__main__':
    main()