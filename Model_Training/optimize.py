import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Function to prepare data for LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_data(df, max_seq_length=5):
    # ✅ Step 1: Ensure latency_before is float
    df['latency_before'] = df['latency_before'].astype(float)

    # ✅ Step 2: Normalize latency_before
    scaler = MinMaxScaler()
    df['latency_before_scaled'] = scaler.fit_transform(df[['latency_before']])
    
    # ✅ Step 3: Group by syscall to create sequences
    sequences = []
    labels = []
    for syscall in df['syscall'].unique():
        syscall_data = df[df['syscall'] == syscall].sort_values('time')
        latency_values = syscall_data['latency_before_scaled'].values
        
        # ✅ Step 4: Create sequences of length max_seq_length
        for i in range(len(latency_values) - max_seq_length):
            sequences.append(latency_values[i:i + max_seq_length])
            labels.append(latency_values[i + max_seq_length])
    
    # ✅ Step 5: Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    print("Sequences shape before reshape:", sequences.shape)

    # ✅ Step 6: Reshape for LSTM [samples, timesteps, features]
    if sequences.shape[0] > 0:
        sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    else:
        print("❌ Not enough data to create sequences. Try reducing max_seq_length or check data.")
        exit()

    return sequences, labels, scaler


# Function to build and train LSTM model
def train_lstm_model(sequences, labels, max_seq_length):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(max_seq_length, 1)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Predict a single value (latency)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    
    # Train the model
    model.fit(sequences, labels, epochs=10, batch_size=16, verbose=1)
    return model

# Function to predict optimized latency
def predict_optimized_latency(df, model, scaler, max_seq_length):
    predictions = []
    
    for syscall in df['syscall'].unique():
        syscall_data = df[df['syscall'] == syscall].sort_values('time')
        latency_values = syscall_data['latency_before_scaled'].values
        
        # Create sequences for prediction
        sequences = []
        for i in range(len(latency_values) - max_seq_length + 1):
            if i == 0:
                seq = [0] * (max_seq_length - len(latency_values)) + list(latency_values[:i + max_seq_length])
            else:
                seq = latency_values[i:i + max_seq_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
        
        # Predict optimized latency
        predicted = model.predict(sequences, verbose=0)
        predicted = scaler.inverse_transform(predicted)  # Denormalize
        
        # Pad predictions to match original data length
        padding = np.zeros(len(latency_values) - len(predicted))
        predicted = np.concatenate([padding, predicted.flatten()])
        predictions.extend(predicted)
    
    # Add predictions to DataFrame
    df['latency_after'] = predictions
    return df

# Main function to run optimization
def main(input_file="syscall_logs.csv", output_file="optimized_logs.csv", max_seq_length=2):
    import pandas as pd

    # Load original file
    df_raw = pd.read_csv(input_file)

    # ✅ Debug Prints: Check what's inside syscall_logs.csv
    print("📄 df_raw HEAD:\n", df_raw.head())
    print("🧩 df_raw COLUMNS:", df_raw.columns)

    # Prepare data
    sequences, labels, scaler = prepare_data(df_raw, max_seq_length)
    model = train_lstm_model(sequences, labels, max_seq_length)

    # Predict optimized latency
    df = predict_optimized_latency(df_raw, model, scaler, max_seq_length)

    # ✅ Add latency_before from raw input
    if 'latency' in df_raw.columns:
        df['latency_before'] = df_raw['latency'].values[:len(df)]
    else:
        print("❌ latency column not found in raw CSV")

    # ✅ Final debug print
    print("✅ Final columns:", df.columns)
    print(df.head())

    # Save file
    df.to_csv(output_file, index=False)
    print("✅ File saved:", output_file)



if __name__ == "__main__":
    main()
