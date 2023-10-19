import csv

def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))

def split_rows_into_three(rows):
    """Splits a list of rows into three lists, where each list contains the rows for a single data point."""
    return [rows[i:i + 3] for i in range(0, len(rows), 3)]

def extract_values(data_points, row_index):
    """Extracts values from a specific row of each data point and converts them to floats."""
    return [[float(value) for value in data_point[row_index][1:]] for data_point in data_points]

def save_data_to_file(filename, data):
    """Saves a list of data to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def main():
    filename = "RCWall_Data/generated_samples.csv"
    rows = open_csv_file(filename)
    data_points = split_rows_into_three(rows)

    # Extract and save the parameter values, displacement values, and y values to separate files
    save_data_to_file("RCWall_Data/InputParameters_values.csv", extract_values(data_points, 0))
    save_data_to_file("RCWall_Data/InputDisplacement_values.csv", extract_values(data_points, 1))
    save_data_to_file("RCWall_Data/OutputShear_values.csv", extract_values(data_points, 2))


# Run file preparation
main()