import pandas as pd

# Define the input file path (replace with your actual path)
input_file = 'RCWall_Data/Dataset_full/OutputPushoverShear.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# # Remove the second column (index 1)
# df.drop(df.columns[1], axis=1, inplace=True)

# # Save the modified DataFrame back to the same file
# df.to_csv(input_file, index=False)

print(f"Successfully removed the second column from '{input_file}' and saved the modified data in the same file.")
