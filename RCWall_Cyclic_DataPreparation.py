import csv
import pandas as pd


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))


def split_rows(rows):
    """Splits a list of rows into six lists, where each list contains the rows for a single data point."""
    return [rows[i:i + 4] for i in range(0, len(rows), 4)]


def extract_values(data_points, row_index):
    """Extracts values from a specific row of each data point and converts them to floats."""
    return [[float(value) for value in data_point[row_index][1:]] for data_point in data_points]


def save_data_to_file(filename, data):
    """Saves a list of data to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


BASE_FILE_PATH = "RCWall_Data"  # Define a constant for the base file path

pushover = True
cyclic = False

if cyclic:
    filename = f"{BASE_FILE_PATH}/RCWall_Dataset_cyclic.csv"
    folder = "Dataset_cyclic"
if pushover:
    filename = f"{BASE_FILE_PATH}/RCWall_Dataset_pushover2.csv"
    folder = "Dataset_pushover"

rows = open_csv_file(filename)
data_points = split_rows(rows)

if cyclic:
    # Extract and save the parameter values, displacement values, and y values to separate files
    # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputParameters_values.csv", extract_values(data_points, 0))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputDisplacement_values.csv", extract_values(data_points, 1))

    # ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputCyclicDisplacement_values.csv", extract_values(data_points, 2))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputCyclicShear_values.csv", extract_values(data_points, 3))

if pushover:
    # Extract and save the parameter values, displacement values, and y values to separate files
    # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputParameters_values.csv", extract_values(data_points, 0))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/InputDisplacement_values.csv", extract_values(data_points, 1))

    # ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputPushoverDisplacement_values.csv", extract_values(data_points, 2))
    save_data_to_file(f"{BASE_FILE_PATH}/{folder}/OutputPushoverShear_values.csv", extract_values(data_points, 3))

