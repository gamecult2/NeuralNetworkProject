import math
import matplotlib.pyplot as plt
from keras import backend as K
from keras.saving.save import load_model
from GenerateCyclicLoading import *
from RCWall_DataProcessing import *


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def plotting(x_data, y_data, x_label, y_label, title, sign, save_fig=False, plotValidation=False):
    plt.rcParams.update({'font.size': 14, "font.family": ["Times New Roman", "Cambria"]})
    # Plot Force vs. Displacement
    plt.figure(figsize=(7, 6), dpi=100)
    # Read test output data to plot
    plt.plot(x_data, y_data, color='blue', linewidth=1.2, label=f'Predicted Loop')
    if plotValidation:
        Test = np.loadtxt(f"DataValidation/{title}.txt", delimiter="\t", unpack="False")
        plt.plot(Test[0, :], Test[1, :], color="black", linewidth=1.0, linestyle="--", label=f'Reference {sign}')
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'size': 14}
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.yticks(fontname='Times New Roman', fontsize=14)
    plt.xticks(fontname='Times New Roman', fontsize=14)
    plt.title(f"Specimen {sign}", fontdict={'fontname': 'Times New Roman', 'fontstyle': 'normal', 'size': 14})
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={'family': 'Times New Roman', 'size': 9})
    if save_fig:
        plt.savefig('DataValidation/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# ---------------------- Load the saved NN model ----------------------------------------
loaded_model = load_model("DNN_Models/DNN_Bi-LSTM(PUSHOVER)", custom_objects={'r_square': r_square})
info = 'BiLSTM'

# loaded_model = load_model("DNN_Models/DNN_LSTM-AE(PUSHOVER)", custom_objects={'r_square': r_square})
# info = 'LSTM-AE'

# loaded_model = load_model("DNN_Models/NN_CNN")
# info = 'CNN'

# ---------------------- Load the scaler for normalization -------------------------------
param_scaler = 'RCWall_Data/Scaler/param_scaler.joblib'
disp_scaler = 'RCWall_Data/Scaler/disp_scaler.joblib'
cyclic_scaler = 'RCWall_Data/Scaler/cyclic_scaler.joblib'
pushover_scaler = 'RCWall_Data/Scaler/pushover_scaler.joblib'


# **********************************************************************
# EXPERIMENTAL DATASET
# **********************************************************************
# ------- Load New data for prediction -------------------------------------------------------------------------------------
def reftest():
    name = 'Thomsen_and_Wallace_RW2'
    sign = 'A'
    parameters_input = np.array((102, 3810, 1220, 190, 42, 434, 448, 0.0300, 0.0029, 0.0920)).reshape(1, -1)
    max_displacement = 86
    return parameters_input, max_displacement, name, sign


def refD():
    name = 'Lefas_SW22'
    sign = 'Ⓓ'
    parameters_input = np.array((65, 650, 1350, 140, 51, 470, 470, 0.0330, 0.0025, 0.1000)).reshape(1, -1)
    max_displacement = 16
    return parameters_input, max_displacement, name, sign


def refE():
    name = 'Zhang_SW1-2'
    sign = 'Ⓔ'
    parameters_input = np.array((120, 1000, 2000, 200, 20, 352, 379, 0.0167, 0.0100, 0.2000)).reshape(1, -1)
    max_displacement = 23
    return parameters_input, max_displacement, name, sign


def refF():
    name = 'Mosoarca_SW1'
    sign = 'Ⓕ'
    parameters_input = np.array((80, 2600, 1250, 80, 50, 386, 386, 0.0101, 0.0101, 0.0110)).reshape(1, -1)
    max_displacement = 26
    return parameters_input, max_displacement, name, sign


timeseries_length = 500
parameters_input, max_displacement, name, sign = refE()
displacement_input = np.linspace(0, max_displacement, 500).reshape(1, -1)[:, :500]


# ------- Normalize New data ------------------------------------------
parameters_input = normalize(parameters_input, scaler_filename=param_scaler, sequence=False, fit=False)
displacement_input = normalize(displacement_input, scaler_filename=disp_scaler, sequence=True, fit=False)

# ------- Predict New data --------------------------------------------
predicted_shear = loaded_model.predict([parameters_input, displacement_input])

# ------- Denormalize New data ------------------------------------------
parameter_values = denormalize(parameters_input, scaler_filename=param_scaler, sequence=False)
DisplacementStep = denormalize(displacement_input, scaler_filename=disp_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear, scaler_filename=pushover_scaler, sequence=True)


plotting(DisplacementStep[-1], predicted_shear[-1], 'Displacement (mm)', 'Base Shear (kN)', name, sign, save_fig=False, plotValidation=True)

# '''
# **********************************************************************
# TESTING DATASET
# **********************************************************************
# Define the number of sample to be used
batch_size = 30000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 500
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, smoothed_data=False)
InParams, InDisp, OutCycShear = returned_data

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutCycShear, test_size=0.15, random_state=42)

# ---------------------- Plotting the results ---------------------------------------------
test_index = 20
new_input_parameters = X_param_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_disp_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]

# Now, you can use the loaded model to make predictions on new data
predicted_shear = loaded_model.predict([new_input_parameters, new_input_displacement])

# ------- Denormalize New data ------------------------------------------
new_input_parameters = denormalize(new_input_parameters, scaler_filename=param_scaler, sequence=False)
new_input_displacement = denormalize(new_input_displacement, scaler_filename=disp_scaler, sequence=True)
real_shear = denormalize(real_shear, scaler_filename=pushover_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear, scaler_filename=pushover_scaler, sequence=True)

test_index = (2, 5, 13)
name = ('Ⓐ', 'Ⓑ', 'Ⓒ')

for k, i in enumerate(test_index):
    plt.figure(figsize=(4*1.1, 3*1.25))
    plt.subplots_adjust(top=0.918, bottom=0.139, left=0.194, right=0.979, hspace=0.2, wspace=0.185)
    plt.plot(new_input_displacement[i], predicted_shear[i], color='red', linewidth=1.0, label=f'{info} prediction')
    plt.plot(new_input_displacement[i], real_shear[i], color="black", linewidth=0.8, linestyle="--", label=f'Reference {name[k]}')
    font_settings = {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 12}
    plt.xlabel('Displacement (mm)', fontdict=font_settings)
    plt.ylabel('Base Shear (kN)', fontdict=font_settings, labelpad=-2)
    plt.title(f'Specimen {name[k]}', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 14})
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    plt.yticks(fontname='Cambria', fontsize=12)
    plt.xticks(fontname='Cambria', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={'family': 'Cambria', 'size': 9})
    plt.savefig(f'CyclicPredicted{name[k]}{info}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
# '''
