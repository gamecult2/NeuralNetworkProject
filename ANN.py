import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from PEERGM import processNGAfile

# Assuming GM_data and DISP_data are your acceleration and displacement data arrays
desc, nPts, dt, Et, GM_data = processNGAfile('GM/RSN1594_CHICHI_TTN051-E.AT2', 1)

#GM_data = np.loadtxt('GM/ARTIFICIAL.dat')
DISP_data = np.loadtxt(f'DataGeneration/Story_Displacement.0.0.out', usecols=1, delimiter=' ')
print('GM_data', len(GM_data))
print('DISP_data', len(DISP_data))
print(nPts)
# Data preprocessing and sequencing
input_sequences = []
output_sequences = []
sequence_length = 10  # Adjust this based on your problem
for i in range(len(GM_data) - sequence_length):
    input_sequences.append(GM_data[i:i+sequence_length])
    output_sequences.append(DISP_data[i+sequence_length])

X = np.array(input_sequences)
y = np.array(output_sequences)

# Splitting into train, validation, and test sets
split_ratio = [0.6, 0.2, 0.2]
split_index = int(split_ratio[0] * len(X))
X_train, X_val, X_test = X[:split_index], X[split_index:split_index*2], X[split_index*2:]
y_train, y_val, y_test = y[:split_index], y[split_index:split_index*2], y[split_index*2:]

# Define and compile the model
model = keras.Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, num_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# Evaluate the model
mse = model.evaluate(X_test, y_test)

# Make predictions
new_acceleration_sequence = ...  # Provide a new acceleration sequence
predicted_displacement_sequence = model.predict(np.array([new_acceleration_sequence]))

# Visualization and analysis
...
