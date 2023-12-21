import csv
import pandas as pd


# Function to apply smoothing to each row using a moving average
def smooth_row(row, alpha=0.35):
    # Use a simple moving average with the specified window size
    smoothed_row = row.ewm(alpha=alpha, adjust=False).mean()
    return smoothed_row


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))


def split_rows(rows):
    """Splits a list of rows into six lists, where each list contains the rows for a single data point."""
    return [rows[i:i + 6] for i in range(0, len(rows), 6)]


def extract_values(data_points, row_index):
    """Extracts values from a specific row of each data point and converts them to floats."""
    return [[float(value) for value in data_point[row_index][1:]] for data_point in data_points]


def save_data_to_file(filename, data):
    """Saves a list of data to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


filename = "RCWall_Data/RCWall_generated_samples(multiAnalysis).csv"
rows = open_csv_file(filename)
data_points = split_rows(rows)
# Extract and save the parameter values, displacement values, and y values to separate files
# ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
save_data_to_file("RCWall_Data/InputParameters_values.csv", extract_values(data_points, 0))
save_data_to_file("RCWall_Data/InputDisplacement_values.csv", extract_values(data_points, 1))
#
# ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
save_data_to_file("RCWall_Data/OutputCyclicDisplacement_values.csv", extract_values(data_points, 2))
save_data_to_file("RCWall_Data/OutputCyclicShear_values.csv", extract_values(data_points, 3))

# ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
save_data_to_file("RCWall_Data/OutputPushoverDisplacement_values.csv", extract_values(data_points, 4))
save_data_to_file("RCWall_Data/OutputPushoverShear_values.csv", extract_values(data_points, 5))

