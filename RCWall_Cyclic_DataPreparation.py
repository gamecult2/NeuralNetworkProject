import csv

input_filename = 'RCWall_Data/generated_samples.csv'
output_dir = 'RCWall_Data/'

# Create lists to store data for each type
parameter_values = []
displacement_values = []
y_values = []

# Read the input CSV file and separate the data
with open(input_filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    current_data = None

    for row in reader:
        if row:
            data_type = row[0]

            if data_type == 'parameter_values':
                current_data = parameter_values
            elif data_type == 'displacement_values':
                current_data = displacement_values
            elif data_type == 'y_values':
                current_data = y_values


# Write the separated data to individual CSV files
with open(output_dir + 'parameters.csv', 'w', newline='') as parameter_file:
    parameter_writer = csv.writer(parameter_file)
    for row in parameter_values:
        parameter_writer.writerow(row)

with open(output_dir + 'displacement_values.csv', 'w', newline='') as displacement_file:
    displacement_writer = csv.writer(displacement_file)
    for row in displacement_values:
        displacement_writer.writerow(row)

with open(output_dir + 'y_values.csv', 'w', newline='') as y_file:
    y_writer = csv.writer(y_file)
    for row in y_values:
        y_writer.writerow(row)
