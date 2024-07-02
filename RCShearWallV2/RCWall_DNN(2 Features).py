from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RCWall_DataProcessing import *

# Define the number of sample to be used
batch_size = 4500  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 500
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=True, smoothed_data=False)
InParams, InDisp, OutCycShear, OutCycDisp, OutPushShear, OutPushDisp, InDisp_min, InDisp_max, OutCycShear_min, OutCycShear_max, OutCycDisp_min, OutCycDisp_max = returned_data
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
print('distributed_parameters', distributed_parameters.shape)

# Concatenate Displacement and Parameters
concatenated_input = concatenate([displacement_input, distributed_parameters], axis=-1)
print('concatenated_input', concatenated_input.shape)
# -------------------- LSTM layer 1 -------------------------------------------------
lstm1 = LSTM(200, return_sequences=True, stateful=False)(concatenated_input)
activation1 = Activation('relu')(lstm1)

# Attention mechanism
attention1 = Attention(use_scale=True)([activation1, activation1])  # Add attention mechanism
attended_representation1 = Multiply()([activation1, attention1])  # Multiply attention weights with LSTM output

# LSTM layer with return_sequences=True
lstm2_input = concatenate([activation1, attended_representation1], axis=-1)  # Concatenate LSTM output with attended representation

# LSTM layer with return_sequences=True
lstm2 = LSTM(200, return_sequences=True, stateful=False)(lstm2_input)
activation2 = Activation('relu')(lstm2)

# Dense layer with 100 units
dense1 = Dense(100)(activation2)
print('dense1', dense1.shape)

# ---------------------- Output layer --------------------------------------------
# Define separate output layers for shear and displacement
shear_output = Flatten()(Dense(1, name='output_shear')(dense1))
print('output_shear', shear_output.shape)

# displacement_output = Flatten()(Dense(1, name='output_displacement')(dense2))
# print('output_displacement', displacement_output.shape)

# ---------------------- Build the model ------------------------------------------
model = Model(inputs=[parameters_input, displacement_input], outputs=[shear_output])

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
    [X_param_train, X_disp_train],  # Input layer (STRUCTURAL PARAMETERS + CYCLIC LOAD)
    [Y_shear_train],  # Output layer (SHEAR + DISPLACEMENT)
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
loss = model.evaluate([X_param_test, X_disp_test], [Y_shear_test, Y_disp_test])
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
    plt.plot(predicted_shear[i], label=f'Predicted Shear - {i + 1}')
    plt.plot(real_shear[i], label=f'Real Shear - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], label=f'Input Displacement - {i + 1}')
    plt.plot(real_displacement[i], label=f'Real Displacement - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
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
    plt.plot(real_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.plot(real_displacement[i], real_shear[i], label=f'True Loop - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], real_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()



# Inverse transform the normalized data before plotting
param_scaler, disp_scaler, cyc_shear_scaler, cyc_disp_scaler, push_shear_scaler, push_disp_scaler = returned_scaler

new_input_parameters = param_scaler.inverse_transform(new_input_parameters)
new_input_displacement = restoration(new_input_displacement, InDisp_min, InDisp_max)
real_shear = restoration(real_shear, OutCycShear_min, OutCycShear_max)
real_displacement = restoration(real_displacement, OutCycDisp_min, OutCycDisp_max)
predicted_shear = restoration(predicted_shear, OutCycShear_min, OutCycShear_max)
# predicted_displacement = cyc_disp_scaler.inverse_transform(predicted_displacement)

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_shear[i], label=f'Predicted Shear load - {i + 1}')
    plt.plot(real_shear[i], label=f'Real Shear load - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()


# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], label=f'Input Displacement - {i + 1}')
    plt.plot(real_displacement[i], label=f'Real Displacement - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
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
    # plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'True Loop - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], real_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
