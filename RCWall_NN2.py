from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
import matplotlib.pyplot as plt
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
parameters_input = Input(shape=(num_features_input_parameters,), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')
print('parameters_input', parameters_input.shape)
print('displacement_input', displacement_input.shape)

distributed_parameters = RepeatVector(sequence_length)(parameters_input)
print('parameters_input2', distributed_parameters.shape)

# Concatenate inputs
concatenated_tensor = concatenate([displacement_input, distributed_parameters], axis=-1)
print('concatenated_tensor', concatenated_tensor.shape)

# CuDNNLSTM layer with return_sequences=True
lstm1 = Bidirectional(LSTM(200, return_sequences=True, stateful=False))(concatenated_tensor)
print('lstm1', lstm1.shape)

# Attention mechanism
attention = Attention(use_scale=True)([lstm1, lstm1])  # Add attention mechanism
attended_representation = Multiply()([lstm1, attention])  # Multiply attention weights with LSTM output

# CuDNNLSTM layer with return_sequences=True
lstm2_input = concatenate([lstm1, attended_representation], axis=-1)  # Concatenate LSTM output with attended representation
lstm2 = Bidirectional(LSTM(200, return_sequences=True, stateful=False))(lstm2_input)
activation2 = Activation('relu')(lstm2)
print('activation2', activation2.shape)

# Dense layer with 100 units
dense1 = Dense(200)(activation2)
print('dense1', dense1.shape)

# ---------------------- Output layer --------------------------------------------
output_shear = Flatten()(Dense(1, name='output_shear')(dense1))
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
    [X_param_train, X_disp_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=400,
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

# ---------------------- Model testing ---------------------------------------------------
# Evaluate the model
loss = model.evaluate([X_param_test, X_disp_test], [Y_shear_test])
print("Test loss:", loss)

# ---------------------- Plotting the results ---------------------------------------------
test_index = 3

new_input_parameters = X_param_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_disp_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]
real_displacement = Y_disp_test[0:test_index]  # Select a single example

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


# Inverse transform the normalized data before plotting
param_scaler, disp_scaler, cyc_shear_scaler, cyc_disp_scaler, push_shear_scaler, push_disp_scaler = returned_scaler

new_input_parameters = param_scaler.inverse_transform(new_input_parameters)
new_input_displacement = disp_scaler.inverse_transform(new_input_displacement)
real_shear = cyc_shear_scaler.inverse_transform(real_shear)
real_displacement = cyc_disp_scaler.inverse_transform(real_displacement)
predicted_shear = cyc_shear_scaler.inverse_transform(predicted_shear)
# predicted_displacement = cyc_disp_scaler.inverse_transform(predicted_displacement)

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
