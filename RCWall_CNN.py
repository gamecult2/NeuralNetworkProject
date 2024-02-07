from keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, concatenate, Flatten, Conv1D, MaxPooling1D, RepeatVector, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K
import keras.callbacks

from RCWall_DataProcessing import *


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Define the number of sample to be used
batch_size = 300000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 500
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, smoothed_data=False)
InParams, InDisp, OutCycShear = returned_data
param_scaler, disp_scaler, cyc_shear_scaler = returned_scaler

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutCycShear, test_size=0.15, random_state=42)

# ---------------------- NN Model Building -------------------------
# Build the neural network model using functional API
parameters_input = Input(shape=(num_features_input_parameters,), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')

distributed_parameters = RepeatVector(sequence_length)(parameters_input)

concatenated_tensor = concatenate([displacement_input, distributed_parameters], axis=-1)

# Apply 1D convolution with ReLU activation (3 layers)
Conv1 = Conv1D(64, 3, padding='same', activation='relu')(concatenated_tensor)
BatchNorm1 = BatchNormalization()(Conv1)
maxpooling1d_1 = MaxPooling1D(pool_size=2)(BatchNorm1)

Conv2 = Conv1D(64, 3, padding='same', activation='relu')(maxpooling1d_1)
BatchNorm2 = BatchNormalization()(Conv2)
maxpooling1d_2 = MaxPooling1D(pool_size=2)(BatchNorm2)

Conv3 = Conv1D(64, 3, padding='same', activation='relu')(maxpooling1d_2)
BatchNorm3 = BatchNormalization()(Conv3)
maxpooling1d_3 = MaxPooling1D(pool_size=2)(BatchNorm3)

# Flatten the output before passing to Dense layers
flatten_layer = Flatten()(maxpooling1d_3)

# Dense layer with 200 units and dropout
dense1 = Dense(200)(flatten_layer)
dropout1 = Dropout(0.2)(dense1)

# Dense layer with 100 units and dropout
dense2 = Flatten()(Dense(100)(dropout1))
dropout2 = Dropout(0.2)(dense2)

# ---------------------- Output layer --------------------------------------------
output_shear = Dense(sequence_length, name='output_shear')(dropout2)
print('output_shear', output_shear.shape)

# ---------------------- Build the model ------------------------------------------
# Build the model
model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)

# ---------------------- Compile the model -----------------------------------------
learning_rate = 0.001
epochs = 500
batch_size = 32
patience = 50

# Define Adam and SGD optimizers
adam_optimizer = Adam(learning_rate)
sgd_optimizer = SGD(learning_rate, momentum=0.9)
model.compile(optimizer=adam_optimizer,
              loss='mse',
              metrics=[MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(), r_square])

# ---------------------- Print Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=patience,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_param_train, X_disp_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.15,
    callbacks=[early_stopping])  # checkpoint_callback or early_stopping

# ---------------------- Save the model ---------------------------------------------
model.save("DNN_CNN(CYCLIC)")  # Save the model after training
# model.save("NN_CNNCyclic(03)")  # Save the model after training

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

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['r_square'], label="Training Accuracy")
plt.plot(history.history['val_r_square'], label="Validation Accuracy")
plt.scatter(best_epoch - 1, history.history['val_r_square'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
plt.xlabel("Epochs")
plt.ylabel("Accuracy R2")
plt.title("Training and Validation Accuracy Over Epochs")
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
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()


new_input_parameters = denormalize(new_input_parameters, param_scaler, sequence=False)
new_input_displacement = denormalize(new_input_displacement, disp_scaler, sequence=True)
real_shear = denormalize(real_shear, cyc_shear_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear, cyc_shear_scaler, sequence=True)

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
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
