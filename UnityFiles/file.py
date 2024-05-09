import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
import pickle

# Load the tumulou.pkl file
data = joblib.load("tumulou.pkl")
print("Shape of data:", data.shape)

processed_data = data[:, 2:3, :, 0]
processed_data = processed_data.reshape(-1, processed_data.shape[2]).transpose()

print('processed_data.shape', processed_data.shape)


df = pd.DataFrame(processed_data)
df.to_csv("tumulou-disp2.csv", index=False)


