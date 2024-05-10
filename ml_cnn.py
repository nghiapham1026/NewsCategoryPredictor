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

'''
Epoch 1/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.3213 - loss: 1.3624 - val_accuracy: 0.3639 - val_loss: 1.3214
Epoch 2/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 19ms/step - accuracy: 0.3482 - loss: 1.3153 - val_accuracy: 0.3639 - val_loss: 1.3131
Epoch 3/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 18ms/step - accuracy: 0.3581 - loss: 1.2899 - val_accuracy: 0.3639 - val_loss: 1.3034
Epoch 4/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 18ms/step - accuracy: 0.3700 - loss: 1.2683 - val_accuracy: 0.3698 - val_loss: 1.2789
Epoch 5/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.4956 - loss: 1.2075 - val_accuracy: 0.5266 - val_loss: 1.2263
Epoch 6/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 18ms/step - accuracy: 0.6988 - loss: 1.1017 - val_accuracy: 0.6450 - val_loss: 1.1177
Epoch 7/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 18ms/step - accuracy: 0.8020 - loss: 0.9257 - val_accuracy: 0.7189 - val_loss: 0.9718
Epoch 8/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 19ms/step - accuracy: 0.8486 - loss: 0.7195 - val_accuracy: 0.7337 - val_loss: 0.8374
Epoch 9/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 18ms/step - accuracy: 0.8574 - loss: 0.5589 - val_accuracy: 0.7337 - val_loss: 0.7448
Epoch 10/10
381/381 ━━━━━━━━━━━━━━━━━━━━ 7s 19ms/step - accuracy: 0.8726 - loss: 0.4169 - val_accuracy: 0.7456 - val_loss: 0.6879

Model Performance on Test Set:
27/27 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7577 - loss: 0.6570
'''