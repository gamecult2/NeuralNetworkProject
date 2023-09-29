import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten
import keras.callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))


# Generate random data for demonstration
np.random.seed(0)
num_samples = 1000
num_parameters = 10
num_features = 1
po_sequence_length = 100

X = np.random.uniform(0, 1, size=(num_samples, num_parameters)).astype(np.float32).round(5)
# y = np.random.uniform(0, 10, size=(num_samples, po_sequence_length))

# Generate linearly spaced displacement values
displacement = np.linspace(0, 10, po_sequence_length)

# Generate multiple curves resembling pushover analysis results
y = np.array([
    np.random.uniform(0.5, 2) * np.sin(np.random.uniform(0.1, 0.5) * displacement) * np.exp(-0.1 * (displacement - np.random.uniform(2, 8))**2)
    for _ in range(num_samples)])


# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the input data (X) to ensure all features are on similar scales
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape the output data to be a 3D tensor (samples, time steps, features)
# y_train = y_train.reshape(y_train.shape[0], -1, 1)
# y_val = y_val.reshape(y_val.shape[0], -1, 1)
# y_test = y_test.reshape(y_test.shape[0], -1, 1)

# Define your LSTM-based neural network architecture using the Functional API
# Layer 1
input_layer = Input(shape=(num_parameters, num_features), name='parameters_input')
lstm_layer1 = LSTM(64, return_sequences=True)(input_layer)
lstm_layer2 = LSTM(32, return_sequences=True)(lstm_layer1)
# flat1 = Flatten()(lstm_layer2)

# Output layer for displacement
output_layer = LSTM(po_sequence_length, return_sequences=True)(lstm_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # , metrics=['mean_absolute_error']
model.summary()

# Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)
# Train the model
history = model.fit(
    X_train,  # Input layer (GMA + STRUCTURAL PARAMETERS)
    y_train,  # Output layer (DISPLACEMENT)
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1  # checkpoint_callback or early_stopping
    )

# Plot the training and validation loss
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Plot a sample of predicted vs. actual load vs. displacement curves
sample_indices = np.random.choice(len(predictions), size=5, replace=False)
for idx in sample_indices:
    plt.plot(predictions[idx], label='Predicted')
    plt.plot(y_test[idx], label='Actual')
    plt.xlabel('Displacement')
    plt.ylabel('Load')
    plt.legend()
    plt.show()
