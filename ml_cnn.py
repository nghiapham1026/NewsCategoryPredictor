import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import preprocess
from sklearn.model_selection import train_test_split

def prepare_data(X, max_length=512):
    # Tokenize text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_length, padding='post')
    return X_pad, tokenizer.word_index

def build_model(hp, input_length):
    model = Sequential()
    model.add(Embedding(
        input_dim=hp.Int('input_dim', min_value=10000, max_value=50000, step=10000), 
        output_dim=hp.Choice('output_dim', values=[64, 128, 256]),
        input_length=input_length
    ))
    model.add(Conv1D(
        filters=hp.Choice('filters', values=[64, 128, 256]),
        kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]),
        activation='relu'
    ))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64), 
        activation='relu'
    ))
    model.add(Dropout(
        rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    ))
    model.add(Dense(4, activation='softmax'))
    
    model.compile(
        optimizer=Adam(
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_model(X, y, input_length):
    tuner = RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_length),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='model_tuning',
        project_name='NewsCategoryTuning'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tuner.search(X_train, y_train, epochs=10, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
    
    best_model = tuner.get_best_models(num_models=1)[0]
    
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data['TITLE'] = data['TITLE'].apply(preprocess.normalize_text)
    
    X, word_index = prepare_data(data['TITLE'])
    y = data['CATEGORY'].astype('category').cat.codes
    
    # Use the actual input length from prepared data
    input_length = X.shape[1]
    
    tune_model(X, y, input_length)

if __name__ == '__main__':
    main()

'''
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
30000             |30000             |input_dim
128               |128               |output_dim
64                |64                |filters
5                 |5                 |kernel_size
128               |128               |dense_units
0.4               |0.4               |dropout_rate
0.00023864        |0.00023864        |learning_rate


Epoch 1/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 7s 52ms/step - accuracy: 0.3393 - loss: 1.3510 - val_accuracy: 0.3787 - val_loss: 1.3082
Epoch 2/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 4s 46ms/step - accuracy: 0.3728 - loss: 1.2851 - val_accuracy: 0.3817 - val_loss: 1.2898
Epoch 3/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 5s 47ms/step - accuracy: 0.4778 - loss: 1.2341 - val_accuracy: 0.5178 - val_loss: 1.2014
Epoch 4/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 5s 47ms/step - accuracy: 0.6964 - loss: 1.0373 - val_accuracy: 0.6450 - val_loss: 0.9557
Epoch 5/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 5s 48ms/step - accuracy: 0.8405 - loss: 0.6349 - val_accuracy: 0.7219 - val_loss: 0.7520
Epoch 6/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 4s 46ms/step - accuracy: 0.8836 - loss: 0.3770 - val_accuracy: 0.7663 - val_loss: 0.6468
Epoch 7/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 5s 47ms/step - accuracy: 0.9618 - loss: 0.2011 - val_accuracy: 0.8136 - val_loss: 0.6109
Epoch 8/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 4s 46ms/step - accuracy: 0.9883 - loss: 0.1085 - val_accuracy: 0.8136 - val_loss: 0.5956
Epoch 9/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 5s 48ms/step - accuracy: 0.9940 - loss: 0.0697 - val_accuracy: 0.8195 - val_loss: 0.5811
Epoch 10/10
96/96 ━━━━━━━━━━━━━━━━━━━━ 5s 48ms/step - accuracy: 0.9988 - loss: 0.0384 - val_accuracy: 0.8254 - val_loss: 0.5840

Trial 1 Complete [00h 00m 50s]
val_accuracy: 0.8254438042640686

Best val_accuracy So Far: 0.8254438042640686
Total elapsed time: 00h 00m 50s
'''