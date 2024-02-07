csv_file_path = 'RCWall_Data/RCWall_generated_samples_cyclic(02).csv'  # Replace 'your_file.csv' with the actual path to your CSV file

with open(csv_file_path, 'r') as file:
    line_count = sum(1 for line in file)

print(f'Number of lines in the CSV file: {line_count}')