import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, RootMeanSquaredError
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D
import keras.callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Activate the GPU
tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Define the number of sample to be used
max_rows = 300

# Read Input files
InputParameters = np.genfromtxt("RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=max_rows)
InputDisplacement = np.genfromtxt("RCWall_data/InputDisplacement_values.csv", delimiter=',', max_rows=max_rows)
# Read Output files
OutputDisplacement = np.genfromtxt("RCWall_data/OutputDisplacement_values.csv", delimiter=',', max_rows=max_rows)
OutputShear = np.genfromtxt("RCWall_data/OutputShear_values.csv", delimiter=',', max_rows=max_rows)

# Normalize the data with separate scalers
param_scaler = MinMaxScaler()
Normalized_InputParameters = param_scaler.fit_transform(InputParameters)

displacement_scaler = StandardScaler()
Normalized_InputDisplacement = displacement_scaler.fit_transform(InputDisplacement.T).T

# Create a StandardScaler object for OutputShear
output_displacement_scaler = StandardScaler()
Normalized_OutputDisplacement = output_displacement_scaler.fit_transform(OutputDisplacement.T).T

output_shear_scaler = StandardScaler()
Normalized_OutputShear = output_shear_scaler.fit_transform(OutputShear.T).T

# Save normalized data to CSV files
np.savetxt("RCWall_Data/Normalized_InputParameters.csv", Normalized_InputParameters, delimiter=',')
np.savetxt("RCWall_Data/Normalized_InputDisplacement.csv", Normalized_InputDisplacement, delimiter=',')
np.savetxt("RCWall_Data/Normalized_OutputDisplacement.csv", Normalized_OutputDisplacement, delimiter=',')
np.savetxt("RCWall_Data/Normalized_OutputShear.csv", Normalized_OutputShear, delimiter=',')

# Organize the Generate data
num_samples, parameters_length = InputParameters.shape
num_samples, sequence_length = InputDisplacement.shape

print('----------------------------------------')
print('InputParameters Shape = ', (InputParameters.shape))
print('InputDisplacement Shape = ', (InputDisplacement.shape))
print('----------------------------------------')

num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)

# Reshape Data
# InputDisplacement = InputDisplacement.reshape(InputDisplacement.shape[0], InputDisplacement.shape[1], num_features)

print('----------------------------------------')
print('InputParameters Shape = ', (InputParameters.shape))
print('InputDisplacement Shape = ', (InputDisplacement.shape))
print('----------------------------------------')

# Split data into training, validation, and testing sets
X_parameter_train, X_parameter_test, X_displacement_train, X_displacement_test, Y_displacement_train, Y_displacement_test, Y_shear_train, Y_shear_test = train_test_split(
    Normalized_InputParameters, Normalized_InputDisplacement, Normalized_OutputDisplacement, Normalized_OutputShear, test_size=0.2, random_state=42
)

# Build the neural network model using functional API
# Layer 1
parameters_input = Input(shape=(parameters_length,), name='parameters_input')
dense_layer = Dense(sequence_length)(parameters_input)  # Dense layer for influencing parameters
flat1 = Flatten()(dense_layer)

# Layer 2
displacement_input = Input(shape=(sequence_length, num_features), name='displacement_input')
# lstm_layer = LSTM(sequence_length, return_sequences=True)(displacement_input)  # Bidirectional LSTM layer
# flat2 = Flatten()(lstm_layer)

# 1D Convolutional Layer for displacement input
conv_layer = Conv1D(sequence_length, kernel_size=3, activation='relu')(displacement_input)
pooled_layer = GlobalMaxPooling1D()(conv_layer)
flat2 = Dense(sequence_length, activation='relu')(Flatten()(pooled_layer))

# Recurrent Layer (LSTM)
# lstm_layer = LSTM(sequence_length, return_sequences=True)(displacement_input)
# flat3 = Dense(sequence_length, activation='relu')(Flatten()(lstm_layer))

# LSTM Autoencoder Model
lstm_encoder = LSTM(32, return_sequences=True)(displacement_input)
lstm_decoder = LSTM(sequence_length, return_sequences=True)(lstm_encoder)
flat2 = Flatten()(lstm_decoder)


# Merge the 2 inputs layer with concatenate LSTM and Dense layers merged = Concatenate()([flat1, flat2])
merged = concatenate([flat1, flat2])

# Output layer for displacement
displacement_output = Dense(sequence_length)(merged)

# Output layer for shear
shear_output = Dense(sequence_length)(merged)

model = Model(inputs=[parameters_input, displacement_input], outputs=[shear_output, displacement_output])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=[MeanSquaredError(name='mse'),
                       MeanAbsoluteError(name='mae')])

# Model summary
model.summary()

# Define the checkpoint callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=5,  # stop training after 5 non-improved training
    verbose=2
)

# Train the model
history = model.fit(
    [X_parameter_train, X_displacement_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train, Y_displacement_train],  # Output layer (SHEAR)
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
# model.save("NN_functionalAPI_DynamicSystem")  # Save the model after training

# Plot the training and validation loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Evaluate the model
loss = model.evaluate([X_parameter_test, X_displacement_test], [Y_shear_test, Y_displacement_test])
print("Test loss:", loss)

test_index = 3
new_parameters = X_parameter_test[0:test_index + 1]  # Select corresponding influencing parameters
new_indisplacement = X_displacement_test[0:test_index + 1]  # Select a single example

real_shear = Y_shear_test[0:test_index + 1]
real_displacement = Y_displacement_test[0:test_index + 1]  # Select a single example

# Predict displacement for the new data
predicted_shear, predicted_displacement = model.predict([new_parameters, new_indisplacement])

# # Restoring the original data from the normalized data
# restored_InputParameters = param_scaler.inverse_transform(InputParameters)
# restored_InputDisplacement = displacement_scaler.inverse_transform(InputDisplacement.T).T
# restored_OutputDisplacement = output_displacement_scaler.inverse_transform(OutputDisplacement.T).T
# restored_OutputShear = output_shear_scaler.inverse_transform(OutputShear.T).T

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_shear[i], label=f'Predicted Shear load - {i + 1}')
    plt.plot(real_shear[i], label=f'Real Shear load - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_displacement[i], label=f'Predicted displacement - {i + 1}')
    plt.plot(real_displacement[i], label=f'Real displacement - {i + 1}')
    plt.plot(new_indisplacement[i], label=f'Input displacement  - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_displacement[i], predicted_shear[i], label=f'Predicted Displacement - {i + 1}')
    plt.plot(real_displacement[i], real_shear[i], label=f'True displacement - {i + 1}')
    plt.plot(new_indisplacement[1:i+1], real_shear[i], label=f'Input displacement - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
