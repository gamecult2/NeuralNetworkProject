import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, RootMeanSquaredError
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add
import keras.callbacks
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
max_rows = 20
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
# ---------------------- Read Data  -------------------------------
# Input files
InputParameters = np.genfromtxt("RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)
InputDisplacement = np.genfromtxt("RCWall_data/InputDisplacement_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)
# Output files
OutputDisplacement = np.genfromtxt("RCWall_data/OutputDisplacement_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)
OutputShear = np.genfromtxt("RCWall_data/OutputShear_values.csv", delimiter=',', max_rows=max_rows).astype(np.float32)

# ---------------------- Data Normalization  ----------------------
# Input Normalization
param_scaler = MinMaxScaler()
Normalized_InputParameters = param_scaler.fit_transform(InputParameters)
displacement_scaler = StandardScaler()
Normalized_InputDisplacement = displacement_scaler.fit_transform(InputDisplacement.T).T
# Output Normalization
output_displacement_scaler = StandardScaler()
Normalized_OutputDisplacement = output_displacement_scaler.fit_transform(OutputDisplacement.T).T
output_shear_scaler = StandardScaler()
Normalized_OutputShear = output_shear_scaler.fit_transform(OutputShear.T).T

# Normalize the data between 0 and 1
# min_value = np.min(InputDisplacement)
# max_value = np.max(InputDisplacement)
# Normalized_InputDisplacement = (InputDisplacement - min_value) / (max_value - min_value)
#
# min_value = np.min(OutputShear)
# max_value = np.max(OutputShear)
# Normalized_OutputShear = (OutputShear - min_value) / (max_value - min_value)
#
# min_value = np.min(OutputDisplacement)
# max_value = np.max(OutputDisplacement)
# Normalized_OutputDisplacement = (OutputDisplacement - min_value) / (max_value - min_value)

# ---------------------- Save Normalized Data --------------------
# Save normalized data to CSV files
np.savetxt("RCWall_Data/Normalized_InputParameters.csv", Normalized_InputParameters, delimiter=',')
np.savetxt("RCWall_Data/Normalized_InputDisplacement.csv", Normalized_InputDisplacement, delimiter=',')

np.savetxt("RCWall_Data/Normalized_OutputDisplacement.csv", Normalized_OutputDisplacement, delimiter=',')
np.savetxt("RCWall_Data/Normalized_OutputShear.csv", Normalized_OutputShear, delimiter=',')
# Organize the Generate data
num_samples, parameters_length = InputParameters.shape
num_samples, sequence_length = InputDisplacement.shape

# ---------------------- Reshape Data --------------------------
# Reshape Data
# Normalized_InputParameters = Normalized_InputParameters.reshape(Normalized_InputParameters.shape[0], Normalized_InputParameters.shape[1], num_features)
# Normalized_InputDisplacement = Normalized_InputDisplacement.reshape(Normalized_InputDisplacement.shape[0], Normalized_InputDisplacement.shape[1], num_features)
print('----------------------------------------')
print('InputParameters Shape = ', (Normalized_InputParameters.shape))
print('InputDisplacement Shape = ', (Normalized_InputDisplacement.shape))
print('----------------------------------------')

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_parameter_train, X_parameter_test, X_displacement_train, X_displacement_test, Y_displacement_train, Y_displacement_test, Y_shear_train, Y_shear_test = train_test_split(
    Normalized_InputParameters, Normalized_InputDisplacement, Normalized_OutputDisplacement, Normalized_OutputShear, test_size=0.2, random_state=42)

# ---------------------- NN Model Building -------------------------
# Build the neural network model using functional API

# Layer 1
parameters_input = Input(shape=(parameters_length,), name='parameters_input')
dense_layer = Dense(32)(parameters_input)  # Dense layer for influencing parameters 32, activation='relu' or sequence_length
flat1 = Flatten()(dense_layer)

# Layer 1 - Replace Dense layer with 1D Convolutional Layer
# parameters_input = Input(shape=(parameters_length, num_features), name='parameters_input')
# conv1d_layer = Conv1D(32, kernel_size=2, activation='relu')(parameters_input)  # 1D Convolutional Layer
# maxpool_layer = MaxPooling1D(pool_size=3)(conv1d_layer)
# flat1 = Flatten()(maxpool_layer)

# Layer 2
displacement_input = Input(shape=(sequence_length, num_features), name='displacement_input')
# lstm_layer = LSTM(sequence_length, return_sequences=True)(displacement_input)  # Bidirectional LSTM layer
# flat2 = Flatten()(lstm_layer)

# 1D Convolutional Layer for displacement input
# conv_layer = Conv1D(sequence_length, kernel_size=3, activation='relu')(displacement_input)
# pooled_layer = GlobalMaxPooling1D()(conv_layer)
# flat2 = Dense(sequence_length, activation='relu')(Flatten()(pooled_layer))

# Recurrent Layer (LSTM)
# lstm_layer = LSTM(sequence_length, return_sequences=True)(displacement_input)
# flat3 = Dense(sequence_length, activation='relu')(Flatten()(lstm_layer))

# LSTM Autoencoder Model
# lstm_encoder = CuDNNLSTM(32, return_sequences=True, stateful=False)(displacement_input)
# lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=False)(lstm_encoder)
# flat2 = Flatten()(lstm_decoder)

# LSTM Autoencoder Model
lstm_encoder = LSTM(32, return_sequences=True)(displacement_input)
lstm_decoder = LSTM(sequence_length, return_sequences=True)(lstm_encoder)
flat2 = Flatten()(lstm_decoder)

# Self-Attention Transformer Layer
# normalized_displacement = LayerNormalization(epsilon=1e-6)(displacement_input)
# self_attention = MultiHeadAttention(num_heads=4, key_dim=sequence_length // 8)(normalized_displacement, normalized_displacement)
# residual_connection = Add()([self_attention, normalized_displacement])

# Feedforward block
# feedforward = LayerNormalization(epsilon=1e-6)(self_attention)
# feedforward = Conv1D(64, kernel_size=4, activation='relu')(feedforward)
# feedforward = Conv1D(32, kernel_size=2)(feedforward)
# flat2 = Flatten()(feedforward)

# LSTM Autoencoder Model
# lstm_encoder = CuDNNLSTM(32, return_sequences=True, stateful=False)(displacement_input)
# lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=True)(lstm_encoder)
# flat2 = Flatten()(lstm_decoder)

# Merge the 2 input layers with concatenate LSTM and Dense layers
merged = concatenate([flat1, flat2])

# ---------------------- Output layer --------------------------------------------
shear_output = Dense(sequence_length, name='shear_output')(merged)  # Shear
displacement_output = Dense(sequence_length, name='displacement_output')(merged)  # Displacement

# ---------------------- Build the model ------------------------------------------
model = Model(inputs=[parameters_input, displacement_input], outputs=[shear_output, displacement_output])

# ---------------------- Compile the model -----------------------------------------
# Define Adam and SGD optimizers
adam_optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer,
              loss='mse',
              metrics=['mse', 'accuracy'])

# ---------------------- Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=10,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_parameter_train, X_displacement_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train, Y_displacement_train],  # Output layer (SHEAR)
    epochs=1000,
    batch_size=16,
    validation_split=0.3,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
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

plt.figure(figsize=(10, 6))
plt.plot(history.history['shear_output_accuracy'], label="Training Accuracy")
plt.plot(history.history['val_shear_output_accuracy'], label="Validation Accuracy")
plt.scatter(best_epoch - 1, history.history['val_shear_output_accuracy'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Over Epochs")
plt.legend()
plt.show()
# # Plot the training and validation accuracy
# plt.plot(history.history['mse'], label="Training MSE")
# plt.plot(history.history['val_mse'], label="Validation MSE")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.title("Training and Validation MSE Over Epochs")
# plt.legend()
# plt.show()

# Evaluate the model
loss = model.evaluate([X_parameter_test, X_displacement_test], [Y_shear_test, Y_displacement_test])
print("Test loss:", loss)


test_index = 3

new_input_parameters = X_parameter_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_displacement_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]
real_displacement = Y_displacement_test[0:test_index]  # Select a single example

# Predict displacement for the new data
predicted_shear, predicted_displacement = model.predict([new_input_parameters, new_input_displacement])


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
plt.figure(figsize=(10, 6))
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
