import os
import csv

def remove_first_column(input_file, output_file=None, preserve_header=True):
    with open(input_file, 'r', newline='') as infile, open(output_file or input_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        if preserve_header:
            header = next(reader)
            writer.writerow(header[1:])  # Write updated header

        for row in reader:
            # Efficiently remove the first column and write the modified row
            writer.writerow(row[1:])

    print(f"Successfully removed the first column from '{input_file}' and saved the modified data to '{output_file or input_file}'.")

def process_csv_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, f"modified_{filename}")
            remove_first_column(input_file, output_file)

if __name__ == '__main__':
    folder_path = 'RCWall_Data/Dataset_full/'  # Replace with your folder path
    process_csv_files_in_folder(folder_path)
