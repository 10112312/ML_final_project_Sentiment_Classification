import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from data_prep import data
from sklearn.model_selection import train_test_split
from emotion_lbl import all_emotions
print('running files')

# Load the pre-trained GloVe embeddings
print('importing embedding document')
path_to_glove = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/glove.42B.300d.txt'
word_vectors = KeyedVectors.load_word2vec_format(path_to_glove, binary=False)

# Define the tokenizer with certain hyperparameter
print('tokenizing')
tokenizer = Tokenizer(num_words=2000, lower=True)

# Fit the tokenizer on the training data
print('fitting the tokenizer')
X_train, X_test = train_test_split(data, test_size=0.85, random_state=42)
X_train, X_val = train_test_split(X_train, test_size=0.8, random_state=42)
tokenizer.fit_on_texts(X_train['text'])

# Convert the text data to numerical sequences
print('converting the text data to numerical sequences')
train_seq = tokenizer.texts_to_sequences(X_train['text'])
val_seq = tokenizer.texts_to_sequences(X_val['text'])
test_seq = tokenizer.texts_to_sequences(X_test['text'])

# Pad the sequences to a fixed length
maxlen = 30
train_pad = keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=maxlen, padding='post')
val_pad = keras.preprocessing.sequence.pad_sequences(val_seq, maxlen=maxlen, padding='post')
test_pad = keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=maxlen, padding='post')

# Create a weight matrix for the embeddings
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

# Create one-hot encodings of the labels
train_labels = X_train['emotion'].values
val_labels = X_val['emotion'].values
test_labels = X_test['emotion'].values

num_classes = len(np.unique(train_labels)) + 1
y_train = to_categorical(train_labels, num_classes)
y_val = to_categorical(val_labels, num_classes)
y_test = to_categorical(test_labels, num_classes)


# Define the model architecture
print('setting up model structure')
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,
                           embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                           trainable=False, input_shape=(maxlen,)),
    keras.layers.Dropout(0.5),
    keras.layers.Conv1D(128, 5, activation='relu', padding='same'),
    keras.layers.MaxPooling1D(5),
    keras.layers.Conv1D(128, 5, activation='relu', padding='same'),
    keras.layers.MaxPooling1D(5),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print('training model')
history = model.fit(train_pad, y_train, validation_data=(val_pad, y_val), epochs=10, batch_size=32)
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# history = model.fit(train_pad, y_train, validation_data=(val_pad, y_val), epochs=10, batch_size=32)

# Evaluate the model on the test set
model.evaluate(test_pad, y_test)

import matplotlib.pyplot as plt

# Plot accuracy and loss during training
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Make a prediction on a new text input
new_text = 'this is impossible'
new_seq = tokenizer.texts_to_sequences([new_text])
new_pad = keras.preprocessing.sequence.pad_sequences(new_seq, maxlen=maxlen, padding='post')
new_pred = model.predict(new_pad)
print(all_emotions[new_pred.argmax()])
print('closing files')