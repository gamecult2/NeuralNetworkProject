import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, RootMeanSquaredError
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda
import keras.callbacks
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os

def scale_data(data, range=(-1, 1)):
    # Reshape the data to 1D
    data_1d = np.reshape(data, [data.shape[0] * data.shape[1], 1])

    # Min-Max scaling
    scaler = MinMaxScaler(feature_range=range)
    data_scaled_1d = scaler.fit_transform(data_1d)

    # Reshape the scaled data back to the original shape
    data_scaled = np.reshape(data_scaled_1d, [data.shape[0], data.shape[1]])

    return data_scaled

# Allocate space for Bidirectional(LSTM)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Activate the GPU
tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Define the number of sample to be used
batch_size = 3000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
num_timeseries = 500
num_features_input_displacement = 1
num_features_input_parameters = 10

# ---------------------- Read Data  -------------------------------
# Input files (Structural Parameters + Cyclic Loading)
InputParameters = np.genfromtxt("RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=batch_size)
InputDisplacement = np.genfromtxt("RCWall_data/InputDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(num_timeseries))
# Output files (Hysteresis Curve)
OutputCyclicDisplacement = np.genfromtxt("RCWall_data/Smoothed/SmoothedOutputCyclicDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(num_timeseries))
OutputCyclicShear = np.genfromtxt("RCWall_data/Smoothed/SmoothedOutputCyclicShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(num_timeseries))
# Output files (Pushover Curve)
# OutputPushoverDisplacement = np.genfromtxt("RCWall_data/OutputPushoverDisplacement_values.csv", delimiter=',', max_rows=max_rows)
# OutputPushoverShear = np.genfromtxt("RCWall_data/OutputPushoverShear_values.csv", delimiter=',', max_rows=max_rows)

# ---------------------- Data Normalization  ----------------------
# Input Normalization (Structural Parameters + Cyclic Loading)
param_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_InputParameters = param_scaler.fit_transform(InputParameters)
displacement_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_InputDisplacement = displacement_scaler.fit_transform(InputDisplacement)
# Output Normalization (Hysteresis Curve)
output_CyclicDisplacement_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_OutputCyclicDisplacement = output_CyclicDisplacement_scaler.fit_transform(OutputCyclicDisplacement.T).T
output_CyclicShear_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_OutputCyclicShear = output_CyclicShear_scaler.fit_transform(OutputCyclicShear.T).T

# # Input Normalization (Structural Parameters + Cyclic Loading)
# param_scaler = MinMaxScaler()
# Normalized_InputParameters = param_scaler.fit_transform(InputParameters)
# displacement_scaler = StandardScaler()
# Normalized_InputDisplacement = displacement_scaler.fit_transform(InputDisplacement.T).T
# # Output Normalization (Hysteresis Curve)
# output_CyclicDisplacement_scaler = StandardScaler()
# Normalized_OutputCyclicDisplacement = output_CyclicDisplacement_scaler.fit_transform(OutputCyclicDisplacement.T).T
# output_CyclicShear_scaler = StandardScaler()
# Normalized_OutputCyclicShear = output_CyclicShear_scaler.fit_transform(OutputCyclicShear.T).T

# ---------------------- Save Normalized Data --------------------
# Save normalized Input data to CSV files
# np.savetxt("RCWall_Data/Normalized/Normalized_InputParameters.csv", Normalized_InputParameters, delimiter=',')
# np.savetxt("RCWall_Data/Normalized/Normalized_InputDisplacement.csv", Normalized_InputDisplacement, delimiter=',')
# # Save normalized Output data to CSV files
# np.savetxt("RCWall_Data/Normalized/Normalized_OutputCyclicDisplacement.csv", Normalized_OutputCyclicDisplacement, delimiter=',')
# np.savetxt("RCWall_Data/Normalized/Normalized_OutputCyclicShear.csv", Normalized_OutputCyclicShear, delimiter=',')

# ---------------------- Reshape Data --------------------------
# Reshape Data
# Normalized_InputParameters = Normalized_InputParameters.reshape(Normalized_InputParameters.shape[0], Normalized_InputParameters.shape[1], num_features)
# Normalized_InputDisplacement = Normalized_InputDisplacement.reshape(Normalized_InputDisplacement.shape[0], Normalized_InputDisplacement.shape[1], num_features)
# Organize the Generate data
num_samples, parameters_length = InputParameters.shape
num_samples, sequence_length = InputDisplacement.shape
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
parameters_input = Input(shape=(num_features_input_parameters, ), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')
print('parameters_input', parameters_input.shape)
print('displacement_input', displacement_input.shape)

parameters_input2 = RepeatVector(num_timeseries)(parameters_input)
print('parameters_input2', parameters_input2.shape)

# Expand dimensions of Displacement to match the shape of Parameters
# expanded_displacement = tf.expand_dims(displacement_input, axis=-1)  # Add a new dimension at the end
# print('expanded_displacement', expanded_displacement.shape)
# Concatenate Displacement and Parameters
# concatenated_tensor = Concatenate(axis=-1)([displacement_input, parameters_input2])
concatenated_tensor = concatenate([displacement_input, parameters_input2], axis=-1)
print('concatenated_tensor', concatenated_tensor.shape)


# CuDNNLSTM layer with return_sequences=True
lstm1 = CuDNNLSTM(200, return_sequences=True, stateful=False)(concatenated_tensor)
activation1 = Activation('relu')(lstm1)
print('activation1', activation1.shape)

# CuDNNLSTM layer with return_sequences=True
lstm2 = CuDNNLSTM(200, return_sequences=True, stateful=False)(activation1)
activation2 = Activation('relu')(lstm2)
print('activation2', activation2.shape)

# Dense layer with 100 units
dense1 = Dense(100)(activation2)
print('dense1', dense1.shape)

# ---------------------- Output layer --------------------------------------------
output_shear = Dense(1, name='output_shear')(dense1)
print('output_shear', output_shear.shape)

# ---------------------- Build the model ------------------------------------------
# model = Model(inputs=[input_parameters, input_displacement], outputs=[output_shear])
# Build the model
model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)
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
    patience=100,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_parameter_train, X_displacement_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=5,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
# ---------------------- Save the model ---------------------------------------------
model.save("NN_DeepLSTM")  # Save the model after training

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
for i in range(test_index):
    plt.plot(real_displacement[i], predicted_shear[i], label=f'Predicted Displacement - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.plot(real_displacement[i], real_shear[i], label=f'True displacement - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], real_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
