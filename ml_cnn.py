import numpy as np
import tensorflow as tf
import preprocess
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data(X, tokenizer, max_length=512):
    # Tokenizing the text data
    encodings = tokenizer(X.tolist(), truncation=True, padding="max_length", max_length=max_length, return_tensors="tf")
    return encodings['input_ids'], encodings['attention_mask']

def build_model(bert_model, learning_rate=5e-5):
    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
    bert_output = bert_model(input_ids, attention_mask=attention_masks)[1]
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(4, activation='softmax')(dropout)
    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(input_ids, attention_masks, y, model, batch_size=8, epochs=10): # Lower batch size if you want to speed up training
    # Ensure the data is in TensorFlow tensor format, if not convert it
    if not isinstance(input_ids, tf.Tensor):
        input_ids = tf.convert_to_tensor(input_ids.values, dtype=tf.int32)
    if not isinstance(attention_masks, tf.Tensor):
        attention_masks = tf.convert_to_tensor(attention_masks.values, dtype=tf.int32)
    if not isinstance(y, tf.Tensor):
        y = tf.convert_to_tensor(y.values, dtype=tf.int32)

    # Convert tensors to numpy for sklearn compatibility
    input_ids_np = input_ids.numpy()
    attention_masks_np = attention_masks.numpy()
    y_np = y.numpy()

    # Splitting the data
    X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
        input_ids_np, attention_masks_np, y_np, test_size=0.2, random_state=69
    )

    # Early stopping to monitor the validation loss and stop training when it starts to increase
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Training the model
    history = model.fit(
        [X_train_ids, X_train_masks], y_train, 
        validation_split=0.1,
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nModel Performance on Test Set:")
    model.evaluate([X_test_ids, X_test_masks], y_test)

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data = preprocess.normalize_text(data)
    data = preprocess.extract_url_features(data)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks = prepare_data(data['TITLE'], tokenizer)
    y = data['CATEGORY'].astype('category').cat.codes

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    model = build_model(bert_model)
    train_model(input_ids, attention_masks, y, model)

if __name__ == '__main__':
    main()
