import csv
import os

# Define output file path
output_file = "RCWall_Data/Combined_Data.csv"

# Iterate through files in the directory
all_files = [f for f in os.listdir("RCWall_Data/") if f.endswith(".csv")]

# Open output file in append mode (to avoid overwriting existing data)
with open(output_file, 'a', newline='') as combined_file:
    writer = csv.writer(combined_file)

    # Skip header row if existing files already have headers
    skip_header = True

    # Loop through each CSV file
    for filename in all_files:
        file_path = os.path.join("RCWall_Data/", filename)

        # Open each file for reading
        with open(file_path, 'r', newline='') as individual_file:
            reader = csv.reader(individual_file)

            # Write rows from each file to the combined file
            writer.writerows(reader)

print(f"CSV files concatenated successfully. Output: {output_file}")
