import numpy as np
import tensorflow as tf
import preprocess
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import plot_metrics

def prepare_data(X, tokenizer, max_length=512):
    # Tokenizing the text data
    encodings = tokenizer(X.tolist(), truncation=True, padding="max_length", max_length=max_length, return_tensors="tf")
    return encodings['input_ids'], encodings['attention_mask']

def build_model(bert_model, learning_rate=5e-5):
    # Define input layers
    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
    
    # Lambda to convert Keras tensors to TensorFlow tensors and specify output shape
    bert_output = tf.keras.layers.Lambda(
        lambda x: bert_model(x[0], attention_mask=x[1])[0],
        output_shape=(512, 768)  # This needs to be set to the output dimensions of the BERT model
    )([input_ids, attention_masks])

    # Extract the [CLS] token's output for classification tasks
    cls_token = tf.keras.layers.Lambda(lambda x: x[:, 0], output_shape=(768,))(bert_output)
    
    # Additional layers on top
    dense = tf.keras.layers.Dense(256, activation='relu')(cls_token)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(4, activation='softmax')(dropout)
    
    # Construct the model
    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

from sklearn.preprocessing import label_binarize

def train_model(input_ids, attention_masks, y, model, batch_size=8, epochs=10):
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

    # Convert labels to one-hot encoding
    classes = np.unique(y_np)
    y_train = label_binarize(y_train, classes=classes)
    y_test_binarized = label_binarize(y_test, classes=classes)

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
    model.evaluate([X_test_ids, X_test_masks], y_test_binarized)

    # Predicting the test set
    y_pred_proba = model.predict([X_test_ids, X_test_masks], batch_size=batch_size)

    # Computing the confusion matrix
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot ROC and Precision-Recall Curves
    plot_metrics.plot_roc_curves("BERT Model", classes, y_test_binarized, y_pred_proba)
    plot_metrics.plot_precision_recall_curves("BERT Model", classes, y_test_binarized, y_pred_proba)

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data = preprocess.normalize_text(data)
    data = preprocess.extract_url_features(data)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    input_ids, attention_masks = prepare_data(data['TITLE'], tokenizer)
    y = data['CATEGORY'].astype('category').cat.codes

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    model = build_model(bert_model)
    train_model(input_ids, attention_masks, y, model)

if __name__ == '__main__':
    main()