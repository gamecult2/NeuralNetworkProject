import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, RootMeanSquaredError
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed
import keras.callbacks
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os

# Allocate space for Bidirectional(LSTM)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Activate the GPU
tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Define the number of sample to be used
max_rows = 3000
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
num_timeseries = 500
# ---------------------- Read Data  -------------------------------
# Input files (Structural Parameters + Cyclic Loading)
InputParameters = np.genfromtxt("RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=max_rows)
InputDisplacement = np.genfromtxt("RCWall_data/InputDisplacement_values.csv", delimiter=',', max_rows=max_rows, usecols=range(num_timeseries))
# Output files (Hysteresis Curve)
OutputCyclicDisplacement = np.genfromtxt("RCWall_data/Smoothed/SmoothedOutputCyclicDisplacement_values.csv", delimiter=',', max_rows=max_rows, usecols=range(num_timeseries))
OutputCyclicShear = np.genfromtxt("RCWall_data/OutputCyclicShear_values.csv", delimiter=',', max_rows=max_rows, usecols=range(num_timeseries))
# Output files (Pushover Curve)
# OutputPushoverDisplacement = np.genfromtxt("RCWall_data/OutputPushoverDisplacement_values.csv", delimiter=',', max_rows=max_rows)
# OutputPushoverShear = np.genfromtxt("RCWall_data/OutputPushoverShear_values.csv", delimiter=',', max_rows=max_rows)

# ---------------------- Data Normalization  ----------------------
# Input Normalization (Structural Parameters + Cyclic Loading)
param_scaler = MinMaxScaler()
Normalized_InputParameters = param_scaler.fit_transform(InputParameters)
displacement_scaler = StandardScaler()
Normalized_InputDisplacement = displacement_scaler.fit_transform(InputDisplacement.T).T
# Output Normalization (Hysteresis Curve)
output_CyclicDisplacement_scaler = StandardScaler()
Normalized_OutputCyclicDisplacement = output_CyclicDisplacement_scaler.fit_transform(OutputCyclicDisplacement.T).T
output_CyclicShear_scaler = StandardScaler()
Normalized_OutputCyclicShear = output_CyclicShear_scaler.fit_transform(OutputCyclicShear.T).T

# Calculate the mean and standard deviation of the original data set.
mean = np.mean(InputParameters)
std = np.std(InputParameters)
normalized_mean = np.mean(Normalized_InputParameters)
normalized_std = np.std(Normalized_InputParameters)

print('Original data set mean:', mean)
print('Original data set standard deviation:', std)
print('Normalized data set mean:', normalized_mean)
print('Normalized data set standard deviation:', normalized_std)

# Check if the means and standard deviations are close to 0 and 1, respectively.
if np.allclose(normalized_mean, 0) and np.allclose(normalized_std, 1):
    print('Data normalization is correct.')
else:
    print('Data normalization is incorrect.')

# ---------------------- Save Normalized Data --------------------
# Save normalized Input data to CSV files
np.savetxt("RCWall_Data/Normalized/Normalized_InputParameters.csv", Normalized_InputParameters, delimiter=',')
np.savetxt("RCWall_Data/Normalized/Normalized_InputDisplacement.csv", Normalized_InputDisplacement, delimiter=',')
# Save normalized Output data to CSV files
np.savetxt("RCWall_Data/Normalized/Normalized_OutputCyclicDisplacement.csv", Normalized_OutputCyclicDisplacement, delimiter=',')
np.savetxt("RCWall_Data/Normalized/Normalized_OutputCyclicShear.csv", Normalized_OutputCyclicShear, delimiter=',')

# Organize the Generate data
num_samples, parameters_length = InputParameters.shape
num_samples, sequence_length = InputDisplacement.shape

# ---------------------- Reshape Data --------------------------
# Reshape Data
# Normalized_InputParameters = Normalized_InputParameters.reshape(Normalized_InputParameters.shape[0], Normalized_InputParameters.shape[1], num_features)
# Normalized_InputDisplacement = Normalized_InputDisplacement.reshape(Normalized_InputDisplacement.shape[0], Normalized_InputDisplacement.shape[1], num_features)
print('----------------------------------------')
print('InputParameters Shape = ', Normalized_InputParameters.shape)
print('InputDisplacement Shape = ', Normalized_InputDisplacement.shape)
print('----------------------------------------')

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_parameter_train, X_parameter_test, X_displacement_train, X_displacement_test, Y_displacement_train, Y_displacement_test, Y_shear_train, Y_shear_test = train_test_split(
    Normalized_InputParameters, Normalized_InputDisplacement, Normalized_OutputCyclicDisplacement, Normalized_OutputCyclicShear, test_size=0.15, random_state=20)

# ---------------------- NN Model Building -------------------------
# Build the neural network model using functional API
# Layer 1
input_parameters = Input(shape=(parameters_length,), name='input_parameters')
dense_layer = Dense(sequence_length)(input_parameters)  # Dense layer for influencing parameters 32, activation='relu' or sequence_length
flat1 = Flatten()(dense_layer)
rflat1 = Reshape((sequence_length, num_features))(flat1)

# Layer 2
input_displacement = Input(shape=(sequence_length, num_features), name='input_displacement')
lstm_layer = CuDNNLSTM(sequence_length, return_sequences=True)(input_displacement)  # Bidirectional LSTM layer
flat2 = Flatten()(lstm_layer)

# Concatenate expanded input parameters and time series displacement
merged_inputs = concatenate([rflat1, lstm_layer])

# LSTM layers to capture temporal dependencies
# lstm_1 = LSTM(32, return_sequences=True)(reshaped_input)
# lstm_1_dropout = Dropout(0.2)(lstm_1)
# lstm_2 = LSTM(sequence_length, return_sequences=True)(lstm_1_dropout)
# lstm_2_dropout = Dropout(0.2)(lstm_2)
# flat = Flatten()(lstm_2_dropout)
# dense_layer = Dense(sequence_length)(flat)

# Reshape to (timesteps, features)
# reshaped_input = Reshape((sequence_length + 32, num_features))(concatenated_input)

# LSTM layer to capture temporal patterns
# lstm_encoder = CuDNNLSTM(32, return_sequences=True, stateful=False)(reshaped_input)
# lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=True)(lstm_encoder)
# flat = Flatten()(lstm_decoder)

# 1D Convolutional Layer for displacement input
# conv_layer = Conv1D(sequence_length, kernel_size=3, activation='relu')(displacement_input)
# pooled_layer = GlobalMaxPooling1D()(conv_layer)
# flat2 = Dense(sequence_length, activation='relu')(Flatten()(pooled_layer))

# Recurrent Layer (LSTM)
# lstm_layer = LSTM(sequence_length, return_sequences=True)(displacement_input)
# flat3 = Dense(sequence_length, activation='relu')(Flatten()(lstm_layer))

# Self-Attention Transformer Layer
# First
# normalized_displacement = LayerNormalization(epsilon=1e-6)(displacement_input)
# self_attention1 = MultiHeadAttention(num_heads=4, key_dim=sequence_length // 8)(normalized_displacement, normalized_displacement)
# residual_connection1 = Add()([self_attention1, normalized_displacement])
# Second
# normalized_residual1 = LayerNormalization(epsilon=1e-6)(residual_connection1)
# self_attention2 = MultiHeadAttention(num_heads=4, key_dim=sequence_length // 8)(normalized_residual1, normalized_residual1)
# residual_connection2 = Add()([self_attention2, normalized_residual1])
# flat2 = Flatten()(residual_connection2)

# Feedforward block
# feedforward = LayerNormalization(epsilon=1e-6)(self_attention)
# feedforward = Conv1D(64, kernel_size=4, activation='relu')(feedforward)
# feedforward = Conv1D(32, kernel_size=2)(feedforward)
# flat2 = Flatten()(feedforward)

# Merge the 2 input layers with concatenate LSTM and Dense layers
# merged = concatenate([dense_layer, flat2])
# reshaped = Reshape((sequence_length + 32, num_features))(merged)

# LSTM Autoencoder Model
lstm_encoder = CuDNNLSTM(32, return_sequences=True)(merged_inputs)
lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=True)(lstm_encoder)
flat2 = Flatten()(lstm_decoder)

# ---------------------- Output layer --------------------------------------------
dense1 = Dense(sequence_length, name='dense1')(flat2)  # Shear
output_shear = Dense(sequence_length, name='output_shear')(dense1)  # Shear
# output_displacement = Dense(sequence_length, name='output_displacement')(flat2)  # Displacement
print('output_shear ', output_shear.shape)

# ---------------------- Build the model ------------------------------------------
model = Model(inputs=[input_parameters, input_displacement], outputs=[output_shear])

# ---------------------- Compile the model -----------------------------------------
# Define Adam and SGD optimizers
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer,
              loss='mse',
              metrics=['mse'])

# ---------------------- Print Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=50,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_parameter_train, X_displacement_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=1000,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
# ---------------------- Save the model ---------------------------------------------
# model.save("NN_functionalAPI_DynamicSystem")  # Save the model after training

# ---------------------- Plot Accuracy and Loss ----------------------------------------
# Find the epoch at which the best performance occurred
best_epoch = np.argmin(history.history['val_loss']) + 1  # +1 because epochs are 1-indexed

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(history.history['shear_output_accuracy'], label="Training Accuracy")
# plt.plot(history.history['val_shear_output_accuracy'], label="Validation Accuracy")
# plt.scatter(best_epoch - 1, history.history['val_shear_output_accuracy'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training and Validation Accuracy Over Epochs")
# plt.legend()
# plt.show()
# # Plot the training and validation accuracy
# plt.plot(history.history['mse'], label="Training MSE")
# plt.plot(history.history['val_mse'], label="Validation MSE")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.title("Training and Validation MSE Over Epochs")
# plt.legend()
# plt.show()

# Evaluate the model
loss = model.evaluate([X_parameter_test, X_displacement_test], [Y_shear_test])
print("Test loss:", loss)


test_index = 3

new_input_parameters = X_parameter_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_displacement_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]
real_displacement = Y_displacement_test[0:test_index]  # Select a single example

# Predict displacement for the new data
predicted_shear = model.predict([new_input_parameters, new_input_displacement])


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
# plt.figure(figsize=(10, 6))
# for i in range(test_index):
#     plt.plot(predicted_displacement[i], label=f'Predicted displacement - {i + 1}')
#     plt.plot(real_displacement[i], label=f'Real displacement - {i + 1}')
#     plt.plot(new_input_displacement[i], label=f'Input displacement  - {i + 1}')
#     plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
#     plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
#     plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
#     plt.yticks(fontname='Cambria', fontsize=14)
#     plt.xticks(fontname='Cambria', fontsize=14)
#     plt.tight_layout()
#     plt.legend()
#     plt.grid()
#     plt.show()
#
# Plot the predicted displacement
# plt.figure(figsize=(10, 6))
# for i in range(test_index):
#     plt.plot(predicted_displacement[i], predicted_shear[i], label=f'Predicted Displacement - {i + 1}')
#     # plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
#     plt.plot(real_displacement[i], real_shear[i], label=f'True displacement - {i + 1}')
#     # plt.plot(new_input_displacement[i, :-1], real_shear[i, 1:], label=f'Input displacement - {i + 1}')
#     plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
#     plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
#     plt.title('Predicted Displacement Time Series', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
#     plt.yticks(fontname='Cambria', fontsize=14)
#     plt.xticks(fontname='Cambria', fontsize=14)
#     plt.tight_layout()
#     plt.legend()
#     plt.grid()
#     plt.show()
