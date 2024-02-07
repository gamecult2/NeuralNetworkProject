from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RCWall_DataProcessing import *


# Define the number of sample to be used
batch_size = 10  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 500
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, smoothed_data=True)
InParams, InDisp, OutCycShear, OutCycDisp, OutPushShear, OutPushDisp = returned_data
param_scaler, disp_scaler, cyc_shear_scaler, cyc_disp_scaler, push_shear_scaler, push_disp_scaler = returned_scaler

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test, Y_disp_train, Y_disp_test, Y_shear2_train, Y_shear2_test, Y_disp2_train, Y_disp2_test = train_test_split(
    InParams, InDisp, OutCycShear, OutCycDisp, OutPushShear, OutPushDisp, test_size=0.15, random_state=20)

# ---------------------- NN Model Building -------------------------
# Build the neural network model using functional API

# Layer 1
parameters_input = Input(shape=(parameters_length,), name='parameters_input')
dense_layer = Dense(32, activation='relu')(parameters_input)  # Dense layer for influencing parameters 32, activation='relu' or sequence_length
flat1 = Flatten()(dense_layer)

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
lstm_encoder = CuDNNLSTM(32, return_sequences=True, stateful=False)(displacement_input)
lstm_decoder = CuDNNLSTM(sequence_length, return_sequences=True)(lstm_encoder)
flat2 = Flatten()(lstm_decoder)

# Merge the 2 inputs layer with concatenate LSTM and Dense layers
merged = Concatenate()([flat1, flat2])

# ---------------------- Output layer --------------------------------------------
# Shear
shear_output = Dense(sequence_length, name='shear_output')(merged)
# Displacement
displacement_output = Dense(sequence_length, name='displacement_output')(merged)

# ---------------------- Build the model ------------------------------------------
model = Model(inputs=[parameters_input, displacement_input], outputs=[shear_output, displacement_output])

# ---------------------- Compile the model -----------------------------------------
optimizer = Adam(learning_rate=0.001, decay=0.0001)
model.compile(optimizer=optimizer,
              loss={'displacement_output': 'mean_squared_error', 'shear_output': 'mean_squared_error'},
              metrics=[MeanSquaredError(name='mse')]) #, MeanAbsoluteError(name='mae')

# ---------------------- Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=50,  # stop training after 5 non-improved training
    verbose=2
)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_parameter_train, X_displacement_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train, Y_displacement_train],  # Output layer (SHEAR)
    epochs=300,
    batch_size=32,
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

# # Plot the predicted displacement
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

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_displacement[i], predicted_shear[i], label=f'Predicted Displacement - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
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
