import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, RootMeanSquaredError
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
from keras.utils import pad_sequences, plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os


def normalize_data(data, range=(-1, 1)):
    # Reshape the data to 1D
    data_1d = np.reshape(data, [data.shape[0] * data.shape[1], 1])
    # Min-Max scaling
    scaler = MinMaxScaler(feature_range=range)
    data_scaled_1d = scaler.fit_transform(data_1d)
    # Reshape the scaled data back to the original shape
    data_scaled = np.reshape(data_scaled_1d, [data.shape[0], data.shape[1]])
    return data_scaled, scaler


def restore_data(scaled_data, scaler):
    # Reshape the scaled data to 1D
    scaled_data_1d = np.reshape(scaled_data, [scaled_data.shape[0] * scaled_data.shape[1], 1])
    # Inverse transform to restore original scale
    restored_data_1d = scaler.inverse_transform(scaled_data_1d)
    # Reshape the restored data back to the original shape
    restored_data = np.reshape(restored_data_1d, [scaled_data.shape[0], scaled_data.shape[1]])
    return restored_data


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
num_features_output = 1

# ---------------------- Read Data  -------------------------------
# Input files (Structural Parameters + Cyclic Loading)
InputParameters = np.genfromtxt("RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=batch_size)
InputDisplacement = np.genfromtxt("RCWall_data/InputDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(num_timeseries))
# Output files (Hysteresis Curve)
OutputCyclicDisplacement = np.genfromtxt("RCWall_data/OutputCyclicDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(num_timeseries))
OutputCyclicShear = np.genfromtxt("RCWall_data/OutputCyclicShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(num_timeseries))
# Output files (Pushover Curve)   Smoothed/Smoothed
OutputPushoverDisplacement = np.genfromtxt("RCWall_data/OutputPushoverDisplacement_values.csv", delimiter=',', max_rows=batch_size)
OutputPushoverShear = np.genfromtxt("RCWall_data/OutputPushoverShear_values.csv", delimiter=',', max_rows=batch_size)

print('OutputCyclicDisplacement', OutputCyclicDisplacement.shape)
print('OutputCyclicShear', OutputCyclicShear.shape)

print('OutputPushoverDisplacement', OutputPushoverDisplacement.shape)
print('OutputPushoverShear', OutputPushoverShear.shape)

# ---------------------- Data Normalization  ----------------------
# Input Normalization (Structural Parameters + Cyclic Loading)
param_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_InputParameters = param_scaler.fit_transform(InputParameters)
displacement_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_InputDisplacement = displacement_scaler.fit_transform(InputDisplacement)
# Output Normalization (Hysteresis Curve)
output_CyclicShear_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_OutputCyclicShear = output_CyclicShear_scaler.fit_transform(OutputCyclicShear)
output_CyclicDisplacement_scaler = MinMaxScaler(feature_range=(-1, 1))
Normalized_OutputCyclicDisplacement = output_CyclicDisplacement_scaler.fit_transform(OutputCyclicDisplacement)

# # ---------------------- Normalize Input Parameters -------------------------------
# scaler_params = StandardScaler()
# Normalized_InputParameters = scaler_params.fit_transform(InputParameters)
#
# # ---------------------- Normalize Input/Output Time Series ------------------------
# scaler_d = StandardScaler()
# scaler_s = StandardScaler()
#
# # Normalize Input Displacement
# Normalized_InputDisplacement = scaler_d.fit_transform(InputDisplacement)
#
# # Normalize Output Cyclic Shear
# Normalized_OutputCyclicShear = scaler_s.fit_transform(OutputCyclicShear)
#
# # Normalize Output Cyclic Displacement
# Normalized_OutputCyclicDisplacement = scaler_d.fit_transform(OutputCyclicDisplacement)

# ---------------------- Save Normalized Data --------------------
# Save normalized Input data to CSV files
np.savetxt("RCWall_Data/Normalized/Normalized_InputParameters.csv", Normalized_InputParameters, delimiter=',')
np.savetxt("RCWall_Data/Normalized/Normalized_InputDisplacement.csv", Normalized_InputDisplacement, delimiter=',')
# Save normalized Output data to CSV files
np.savetxt("RCWall_Data/Normalized/Normalized_OutputCyclicShear.csv", Normalized_OutputCyclicShear, delimiter=',')
np.savetxt("RCWall_Data/Normalized/Normalized_OutputCyclicDisplacement.csv", Normalized_OutputCyclicDisplacement, delimiter=',')

# Organize the Generate data
num_samples, parameters_length = InputParameters.shape
num_samples, sequence_length = InputDisplacement.shape
print('----------------------------------------')
print('InputParameters Shape = ', Normalized_InputParameters.shape)
print('InputDisplacement Shape = ', Normalized_InputDisplacement.shape)
print('----------------------------------------')
# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_parameter_train, X_parameter_test, X_displacement_train, X_displacement_test, Y_shear_train, Y_shear_test, Y_displacement_train, Y_displacement_test = train_test_split(
    Normalized_InputParameters, Normalized_InputDisplacement, Normalized_OutputCyclicShear, Normalized_OutputCyclicDisplacement, test_size=0.15, random_state=42)

# ---------------------- NN Model Building -------------------------
# Build the neural network model using functional API
parameters_input = Input(shape=(num_features_input_parameters,), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')
print('parameters_input', parameters_input.shape)
print('displacement_input', displacement_input.shape)
lower_input = Dense(3)(parameters_input)

distributed_parameters = RepeatVector(num_timeseries)(parameters_input)
print('distributed_parameters', distributed_parameters.shape)

# Concatenate Displacement and Parameters
concatenated_input = concatenate([displacement_input, distributed_parameters], axis=-1)
print('concatenated_input', concatenated_input.shape)
# -------------------- LSTM layer 1 -------------------------------------------------
lstm1 = Bidirectional(LSTM(200, return_sequences=True, stateful=False))(concatenated_input)
activation1 = Activation('relu')(lstm1)

# Attention mechanism
attention1 = Attention(use_scale=True)([activation1, activation1])  # Add attention mechanism
attended_representation1 = Multiply()([activation1, attention1])  # Multiply attention weights with LSTM output

# LSTM layer with return_sequences=True
lstm2_input = concatenate([activation1, attended_representation1], axis=-1)  # Concatenate LSTM output with attended representation

# LSTM layer with return_sequences=True
lstm2 = Bidirectional(LSTM(200, return_sequences=True, stateful=False))(lstm2_input)
activation2 = Activation('relu')(lstm2)

# -------------------- LSTM layer 2 -------------------------------------------------
lstm11 = Bidirectional(LSTM(200, return_sequences=True, stateful=False))(concatenated_input)
activation11 = Activation('relu')(lstm11)

# Attention mechanism
attention2 = Attention(use_scale=True)([activation11, activation11])  # Add attention mechanism
attended_representation2 = Multiply()([activation11, attention2])  # Multiply attention weights with LSTM output

# LSTM layer with return_sequences=True
lstm22_input = concatenate([activation11, attended_representation2], axis=-1)  # Concatenate LSTM output with attended representation

# LSTM layer with return_sequences=True
lstm22 = Bidirectional(LSTM(200, return_sequences=True, stateful=False))(lstm22_input)
activation22 = Activation('relu')(lstm22)

# Dense layer with 100 units
dense1 = Dense(100)(activation2)
dense2 = Dense(100)(activation22)
print('dense1', dense1.shape)
print('dense2', dense2.shape)

# ---------------------- Output layer --------------------------------------------
# Define separate output layers for shear and displacement
shear_output = Flatten()(Dense(1, name='output_shear')(dense1))
displacement_output = Flatten()(Dense(1, name='output_displacement')(dense2))
print('output_shear', shear_output.shape)
print('output_displacement', displacement_output.shape)

# ---------------------- Build the model ------------------------------------------
model = Model(inputs=[parameters_input, displacement_input], outputs=[shear_output, displacement_output])

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
    [X_parameter_train, X_displacement_train],  # Input layer (STRUCTURAL PARAMETERS + CYCLIC LOAD)
    [Y_shear_train, Y_displacement_train],  # Output layer (SHEAR + DISPLACEMENT)
    epochs=600,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
# ---------------------- Save the model ---------------------------------------------
# model.save("NN_DeepLSTM")  # Save the model after training

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

# Evaluate the model
loss = model.evaluate([X_parameter_test, X_displacement_test], [Y_shear_test, Y_displacement_test])
print("Test loss:", loss)

# ---------------------- Plotting the results ---------------------------------------------
test_index = 5  # Number of sample to plot

new_input_parameters = X_parameter_test[0:test_index]  # Select influencing parameters
new_input_displacement = X_displacement_test[0:test_index]  # Select input displacement

real_shear = Y_shear_test[0:test_index]  # Select output shear
real_displacement = Y_displacement_test[0:test_index]  # Select output displacement

# Predict displacement for the new data
predicted_shear, predicted_displacement = model.predict([new_input_parameters, new_input_displacement])

# Inverse transform the normalized data before plotting
new_input_parameters = param_scaler.inverse_transform(new_input_parameters)
new_input_displacement = displacement_scaler.inverse_transform(new_input_displacement)
real_shear = output_CyclicShear_scaler.inverse_transform(real_shear)
real_displacement = output_CyclicDisplacement_scaler.inverse_transform(real_displacement)
predicted_shear = output_CyclicShear_scaler.inverse_transform(predicted_shear)
predicted_displacement = output_CyclicDisplacement_scaler.inverse_transform(predicted_displacement)

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
    plt.plot(new_input_displacement[i], label=f'Input displacement  - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
#
# Plot the predicted displacement
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_displacement[i], predicted_shear[i], label=f'Predicted Displacement - {i + 1}')
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
