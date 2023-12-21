import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Function to apply smoothing to each row using a moving average
def smooth_row(row, alpha=0.8, window_size=3):
    # Exponential smoothing
    smoothed_row_exp = row.ewm(alpha=alpha, adjust=False).mean()

    # Moving average applied to the exponentially smoothed data
    smoothed_row = smoothed_row_exp.rolling(window=window_size, min_periods=1, center=True).mean()

    return smoothed_row_exp


# ----------------------- Read Data --------------------------------------------------------------------------
#  Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement)
OutputCyclicDisplacement_values = pd.read_csv('RCWall_data/OutputCyclicDisplacement_values.csv')
OutputCyclicShear_values = pd.read_csv('RCWall_data/OutputCyclicShear_values.csv')
#  Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement)
OutputPushoverDisplacement_values = pd.read_csv('RCWall_data/OutputPushoverDisplacement_values.csv')
OutputPushoverShear_values = pd.read_csv('RCWall_data/OutputPushoverShear_values.csv')

# ----------------------- Apply smoothing of the data --------------------------------------------------------
#  Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement)
smoothed_OutputCyclicDisplacement_values = OutputCyclicDisplacement_values.apply(smooth_row, axis=1)
smoothed_OutputCyclicShear_values = OutputCyclicShear_values.apply(smooth_row, axis=1)
#  Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement)
smoothed_OutputPushoverDisplacement_values = OutputPushoverDisplacement_values.apply(smooth_row, axis=1)
smoothed_smoothed_OutputPushoverShear_values = OutputPushoverShear_values.apply(smooth_row, axis=1)

# ----------------------- Save the Data ----------------------------------------------------------------------
#  Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement)
smoothed_OutputCyclicDisplacement_values.to_csv('RCWall_data/Smoothed/SmoothedOutputCyclicDisplacement_values.csv', index=False)
smoothed_OutputCyclicShear_values.to_csv('RCWall_data/Smoothed/SmoothedOutputCyclicShear_values.csv', index=False)
#  Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement)
smoothed_OutputPushoverDisplacement_values.to_csv('RCWall_data/Smoothed/SmoothedOutputPushoverDisplacement_values.csv', index=False)
smoothed_smoothed_OutputPushoverShear_values.to_csv('RCWall_data/Smoothed/SmoothedOutputPushoverShear_values.csv', index=False)


# Read data from CSV files
# output_cyclic_shear_values = pd.read_csv('RCWall_data/OutputCyclicShear_values.csv')
# smoothed_output_cyclic_shear_values = pd.read_csv('RCWall_data/Smoothed/SmoothedOutputCyclicShear_values.csv')

# Load input and output cyclic displacement values directly into NumPy arrays
x1 = np.genfromtxt('RCWall_data/OutputCyclicDisplacement_values.csv', delimiter=',', dtype=np.float16)
y1 = np.genfromtxt('RCWall_data/OutputCyclicShear_values.csv', delimiter=',', dtype=np.float16)
x2 = np.genfromtxt('RCWall_data/Smoothed/SmoothedOutputCyclicDisplacement_values.csv', delimiter=',', dtype=np.float16)
y2 = np.genfromtxt('RCWall_data/Smoothed/SmoothedOutputCyclicShear_values.csv', delimiter=',', dtype=np.float16)


# Load only the 1st row using pandas
# row1 = (pd.read_csv('element_output.csv', delimiter=',', header=None, nrows=1))
# row2 = (pd.read_csv('element_output.csv', delimiter=',', header=None, skiprows=1, nrows=1))
#
# print(row1)
# print(row2)
#
# # smoothed_column1 = df.apply(smooth_row, axis=1)
# #
# # # Apply smoothing to each column
# smoothed_column1 = row1.apply(smooth_row, axis=1)
# smoothed_column2 = row2.apply(smooth_row, axis=1)


# Plot a loop of the first 10 rows from OutputCyclicShear_values.csv and SmoothedOutputCyclicShear_values.csv
for index in range(10, 20):
    fig = plt.subplots(figsize=(10, 6))
    # plt.plot(column1, column2, label=f'Original Row {index + 1}', linestyle='--')
    plt.plot(x1[index], y1[index], label=f'Original Row {index + 1}', linestyle='--')
    plt.plot(x2[index], y2[index], label=f'Smoothed Row {index + 1}')
    plt.title('First 10 Original and Smoothed Rows')
    plt.xlabel('Data Point')
    plt.ylabel('Shear Value')
    plt.legend()
    plt.show()

# '''