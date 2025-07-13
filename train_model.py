import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dummy dataset
texts = [
    "You are awesome", "I hate you", "You are so stupid", "Have a nice day",
    "You are a loser", "This is terrible", "You are kind", "I will kill you"
]
labels = [
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0]
]

MAX_LEN = 150
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

X = padded
y = np.array(labels)

model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=MAX_LEN),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dense(6, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=5, verbose=1)

model.save("model.h5")
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)