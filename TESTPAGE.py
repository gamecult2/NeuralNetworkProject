import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Generate random data for demonstration purposes
np.random.seed(0)
num_samples = 1000
input_features = 10
output_features = 100

# Generate random input parameters (geometry and material properties)
X = np.random.rand(num_samples, input_features)

# Generate random output data (time history of shear force vs. displacement)
Y = np.random.rand(num_samples, output_features)

# Split the data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape Y_train, Y_val, and Y_test to match the expected output shape
Y_train = Y_train.reshape(-1, output_features, 1)
Y_val = Y_val.reshape(-1, output_features, 1)
Y_test = Y_test.reshape(-1, output_features, 1)

# Define your RNN model with a Bidirectional LSTM layer
model = Sequential()
model.add(Bidirectional(LSTM(units=output_features), input_shape=(input_features, 1)))  # Bidirectional LSTM with return_sequences=True
model.add(Dense(units=output_features))  # Output layer for each time step

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=32)

# Predict sequences for the test set
predicted_sequences = model.predict(X_test)

# Plot 5 random examples
num_samples_to_plot = 5

# Randomly choose sample indices to plot
sample_indices = np.random.choice(len(Y_test), num_samples_to_plot, replace=False)

# Create subplots
fig, axes = plt.subplots(num_samples_to_plot, figsize=(12, 10))

for i, sample_index in enumerate(sample_indices):
    # Extract the real and predicted sequences for the sample
    real_sequence = Y_test[sample_index].flatten()
    predicted_sequence = predicted_sequences[sample_index].flatten()

    # Create a time axis for the plot (assuming 100 time steps)
    time_steps = np.arange(output_features)

    # Plot the real and predicted sequences on the corresponding subplot
    axes[i].plot(time_steps, real_sequence, label='Real Sequence', marker='o', linestyle='-')
    axes[i].plot(time_steps, predicted_sequence, label='Predicted Sequence', marker='x', linestyle='--')
    axes[i].set_xlabel('Time Step')
    axes[i].set_ylabel('Shear Force')
    axes[i].set_title(f'Example {i+1}: Real vs. Predicted')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
