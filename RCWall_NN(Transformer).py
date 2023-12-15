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
max_rows = 400
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
num_timeseries = 500
# ---------------------- Read Data  -------------------------------
# Input files (Structural Parameters + Cyclic Loading)
InputParameters = np.genfromtxt("RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)
InputDisplacement = np.genfromtxt("RCWall_data/InputDisplacement_values.csv", delimiter=',', max_rows=max_rows, usecols=range(num_timeseries)).astype(np.float32)
# Output files (Hysteresis Curve)
OutputCyclicDisplacement = np.genfromtxt("RCWall_data/Smoothed/SmoothedOutputCyclicDisplacement_values.csv", delimiter=',', max_rows=max_rows, usecols=range(num_timeseries)).astype(np.float32)
OutputCyclicShear = np.genfromtxt("RCWall_data/Smoothed/SmoothedOutputCyclicShear_values.csv", delimiter=',', max_rows=max_rows, usecols=range(num_timeseries)).astype(np.float32)
# Output files (Pushover Curve)
# OutputPushoverDisplacement = np.genfromtxt("RCWall_data/OutputPushoverDisplacement_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)
# OutputPushoverShear = np.genfromtxt("RCWall_data/OutputPushoverShear_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)

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

# Layer 2
input_displacement = Input(shape=(sequence_length, num_features), name='input_displacement')
# lstm_encoder = CuDNNLSTM(32, return_sequences=True)(input_displacement)
# lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=True)(lstm_encoder)
# flat2 = Flatten()(lstm_decoder)

# 1D Convolutional Layer for displacement input
conv_layer = Conv1D(sequence_length, kernel_size=3, activation='relu')(input_displacement)
pooled_layer = GlobalMaxPooling1D()(conv_layer)
flat2 = Dense(sequence_length, activation='relu')(Flatten()(pooled_layer))

# Concatenate expanded input parameters and time series displacement
merged_inputs = concatenate([flat1, flat2])
merged_inputs = Reshape((merged_inputs.shape[1], num_features))(merged_inputs)

lstm_encoder = CuDNNLSTM(32, return_sequences=True)(merged_inputs)
lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=True)(lstm_encoder)
dense = Flatten()(lstm_decoder)


dense1 = Dense(128, name='dense1')(dense)  # Shear
# ---------------------- Output layer --------------------------------------------

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
    patience=20,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_parameter_train, X_displacement_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=500,
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
