import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ============================
# 1. DATA LOADING & PREPROCESSING
# ============================

def load_dataset(filename):
    """Load system call sequences from a text file."""
    with open(filename, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    return sequences

# Load dataset
dataset = load_dataset("001_NORMAL_Flight.txt")
print("Number of sequences loaded:", len(dataset))

# Tokenize system call sequences (limit vocab size to 500)
tokenizer = Tokenizer(num_words=500, lower=True, split=' ')
tokenizer.fit_on_texts(dataset)
vocab_size = min(len(tokenizer.word_index) + 1, 500)  # Limit vocab size
print("Vocabulary size:", vocab_size)

# Convert sequences into tokenized input/output pairs
input_sequences = []
labels = []
for seq in dataset:
    token_list = tokenizer.texts_to_sequences([seq])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i])
        labels.append(token_list[i])

# Determine max sequence length & pad sequences
max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

# Convert labels into numpy array (no one-hot encoding)
labels = np.array(labels)

# ============================
# 2. MODEL BUILDING & TRAINING
# ============================

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=max_seq_length),  # Smaller embedding dim
    Masking(mask_value=0.0),
    LSTM(64),  # Reduced LSTM size
    Dense(vocab_size, activation='softmax')
])

# Use sparse_categorical_crossentropy (no need for one-hot encoding)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model (reduced batch size & epochs)
model.fit(input_sequences, labels, epochs=10, batch_size=16, verbose=1)

# ============================
# 3. PREDICTION FUNCTION
# ============================

def predict_next_syscall(sequence, model, tokenizer, max_seq_length):
    """Predict the next system call based on an input sequence."""
    token_seq = tokenizer.texts_to_sequences([" ".join(sequence)])[0]
    token_seq = pad_sequences([token_seq], maxlen=max_seq_length, padding='pre')
    
    prediction = model.predict(token_seq, verbose=0)
    predicted_token = np.argmax(prediction, axis=1)[0]
    
    inv_map = {v: k for k, v in tokenizer.word_index.items()}
    return inv_map.get(predicted_token, "Unknown")

# Example Prediction
test_sequence = ["open", "read", "write"]
predicted_call = predict_next_syscall(test_sequence, model, tokenizer, max_seq_length)
print(f"Predicted Next Call: {predicted_call}")

