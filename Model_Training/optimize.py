import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Function to prepare data for LSTM
def prepare_data(df, max_seq_length=5):
    # Normalize latency_before
    scaler = MinMaxScaler()
    df['latency_before_scaled'] = scaler.fit_transform(df[['latency_before']])
    
    # Group by syscall to create sequences
    sequences = []
    labels = []
    for syscall in df['syscall'].unique():
        syscall_data = df[df['syscall'] == syscall].sort_values('time')
        latency_values = syscall_data['latency_before_scaled'].values
        
        # Create sequences of length max_seq_length
        for i in range(len(latency_values) - max_seq_length):
            sequences.append(latency_values[i:i + max_seq_length])
            labels.append(latency_values[i + max_seq_length])
    
    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Reshape for LSTM [samples, timesteps, features]
    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    
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
def main(input_file="syscall_logs.csv", output_file="optimized_logs.csv", max_seq_length=5):
    # Load data from Module 1
    df = pd.read_csv(input_file)
    
    # Prepare data for LSTM
    sequences, labels, scaler = prepare_data(df, max_seq_length)
    
    # Train LSTM model
    model = train_lstm_model(sequences, labels, max_seq_length)
    
    # Predict optimized latency
    df = predict_optimized_latency(df, model, scaler, max_seq_length)
    
    # Save optimized data
    df.to_csv(output_file, index=False)
    print(f"Optimized data saved to {output_file}")

if __name__ == "__main__":
    main()
