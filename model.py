import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained GloVe word embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Load the data
texts = [...]  # List of strings
labels = [...]  # List of labels

# Tokenize the text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences
maxlen = 100
data = pad_sequences(sequences, maxlen=maxlen)

# Create the embedding matrix
word_index = tokenizer.word_index
embedding_dim = 100
num_words = min(10000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the model
model = keras.models.Sequential([
    keras.layers.Embedding(num_words, embedding_dim,
                           embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                           trainable=False),
    keras.layers.Dropout(0.5),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.MaxPooling1D(5),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.MaxPooling1D(5),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=10, validation_split=0.2)
