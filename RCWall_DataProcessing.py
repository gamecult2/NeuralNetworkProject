import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
# from RCWall_Cyclic_Parameters import *
import joblib
from pathlib import Path  # For path handling


# Allocate space for Bidirectional(LSTM)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Activate the GPU
tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))


def normalize(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, fit=False, save_scaler_path=None):
    # Check NOT fit (First Normalization) Then must load a scaler or scaler_filename
    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    if scaler is None:
        if scaler_filename:
            # Load the scaler if a filename is provided
            if not os.path.exists(scaler_filename):
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
            scaler = joblib.load(scaler_filename)
        else:
            # Create a new scaler if neither scaler nor scaler filename is provided
            scaler = MinMaxScaler(feature_range=range)

    if sequence:
        data_reshaped = data.reshape(-1, 1)

        if fit:
            data_scaled = scaler.fit_transform(data_reshaped)
            # Print the minimum and maximum values of the scaler
            print("Min value of the scaler:", scaler.data_min_)
            print("Max value of the scaler:", scaler.data_max_)
        else:
            data_scaled = scaler.transform(data_reshaped)

        data_scaled = data_scaled.reshape(data.shape)

    else:
        if fit:
            data_scaled = scaler.fit_transform(data)
            # Print the minimum and maximum values of the scaler
            print("Min value of the scaler:", scaler.data_min_)
            print("Max value of the scaler:", scaler.data_max_)
        else:
            data_scaled = scaler.transform(data)

    # Save the scaler if a path is provided
    if save_scaler_path and fit:
        joblib.dump(scaler, save_scaler_path)

    if scaler_filename:
        return data_scaled
    else:
        return data_scaled, scaler


def denormalize(data_scaled, scaler=None, scaler_filename=None, sequence=False):
    if scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for denormalization.")

    if sequence:
        data_reshaped = data_scaled.reshape(-1, 1)

        if scaler_filename:
            scaler = joblib.load(scaler_filename)

        data_restored_1d = scaler.inverse_transform(data_reshaped)
        data_restored = data_restored_1d.reshape(data_scaled.shape)

    else:
        if scaler_filename:
            scaler = joblib.load(scaler_filename)

        data_restored = scaler.inverse_transform(data_scaled)

    return data_restored


def read_data(batch_size=100, sequence_length=500, normalize_data=True, save_normalized_data=False, smoothed_data=False):
    # ---------------------- Read Data  -------------------------------
    data_folder = Path("RCWall_Data")  # Base data folder
    if smoothed_data == True:
        data_folder = data_folder / "Smoothed"

    # Read input and output data
    InParams = np.genfromtxt(data_folder / "Dataset_pushover/InputParameters_values.csv", delimiter=',', max_rows=batch_size)
    InDisp = np.genfromtxt(data_folder / "Dataset_pushover/InputDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # OutCycShear = np.genfromtxt(data_folder / "Dataset_cyclic/OutputCyclicShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    OutPushShear = np.genfromtxt(data_folder / "Dataset_pushover/OutputPushoverShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))

    if normalize_data:
        # Normalize data and save scalers
        NormInParams, param_scaler = normalize(InParams, sequence=False, scaler_filename=None, fit=True, save_scaler_path=data_folder / "Scaler/param_scaler.joblib")
        NormInDisp, disp_scaler = normalize(InDisp, sequence=True, scaler_filename=None, fit=True, save_scaler_path=data_folder / "Scaler/disp_scaler.joblib")
        # NormOutCycShear, cyc_shear_scaler = normalize(OutCycShear, sequence=True, scaler_filename=None, fit=True, save_scaler_path=data_folder / "Scaler/cyc_shear_scaler.joblib")
        NormOutPushShear, cyc_pushover_scaler = normalize(OutPushShear, sequence=True, scaler_filename=None, fit=True, save_scaler_path=data_folder / "Scaler/cyc_pushover_scaler.joblib")

        if save_normalized_data:
            # Save normalized data to CSV files
            np.savetxt(data_folder / "Normalized/InputParameters.csv", NormInParams, delimiter=',')
            np.savetxt(data_folder / "Normalized/InputDisplacement.csv", NormInDisp, delimiter=',')
            # np.savetxt(data_folder / "Normalized/OutputCyclicShear.csv", NormOutCycShear, delimiter=',')
            np.savetxt(data_folder / "Normalized/OutputPushoverShear.csv", NormOutPushShear, delimiter=',')

        # print('InputParameters Shape = ', InParams.shape)
        # print('InputDisplacement Shape = ', InDisp.shape)
        return (NormInParams, NormInDisp, NormOutPushShear), \
               (param_scaler, disp_scaler, cyc_pushover_scaler)

    else:
        # print("Raw Data Loaded")
        return InParams, InDisp, OutPushShear
