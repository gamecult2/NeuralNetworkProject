import os
import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def calculate_metrics(target, output):
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    r2 = r2_score(target, output)
    R, p = pearsonr(target, output)
    return mse, mae, R


def read_text_files(base_dir, building_numbers=None, floors_numbers=None, waves=None, directions=None, data_types=None):
    # Allow for optional building number list
    if not floors_numbers:
        floors_numbers = ['6', '7', '6', '7', '7', '4', '2', '2', '5', '5', '10', '4', '3', '5', '5', '8', '7']
    if not building_numbers:
        building_numbers = ['03', '21', '24', '27', '34', '40', '45', '47', '60', '61', '70-2', '71', '73', '75', '82', '86-3', '8990']
    if not directions:
        directions = ['x', 'y']
    if not data_types:
        data_types = ['A', 'U']

    results = {}  # Dictionary to store results (building_dir -> {data_type -> {'target': [], 'output': [], 'metrics': {}}})
    building_results = {}
    # Loop through each building
    # for building_number in building_numbers:
    for building_number, floor_count in zip(building_numbers, floors_numbers):
        for direction in directions:
            building_dir = f'tongji_{building_number}_elastic_{direction.lower()}'
            # Loop through each floor
            # for floor_number in floor_numbers:
            for floor_number in range(2, int(floor_count) + 1):  # Loop up to floor_count (inclusive)
                floor_dir = f'layer_{floor_number}'
                # print(floor_dir)
                # Loop through Displacement (A) and Acceleration (U) folders
                for data_type in data_types:
                    data_type = f'{data_type}'
                    wave = f'{waves}'
                    target_dir = os.path.join(base_dir, building_dir, floor_dir, wave, data_type)
                    # Check if directory exists
                    if os.path.exists(target_dir):
                        txt_files = glob.glob(os.path.join(target_dir, '*.txt'))

                        for txt_file in txt_files:
                            with open(txt_file, 'r') as file:
                                lines = file.readlines()
                                # print(txt_file)
                                if len(lines) >= 2:
                                    target_data = np.array([float(x) for x in lines[0].split()])
                                    output_data = np.array([float(x) for x in lines[1].split()])
                                    mse, mae, r2 = calculate_metrics(target_data, output_data)

                                    # Store results
                                    key = (building_number, direction, floor_number, data_type)
                                    if key not in results:
                                        results[key] = {'mse': [], 'mae': [], 'r2': []}

                                    results[key]['mse'].append(mse)
                                    results[key]['mae'].append(mae)
                                    results[key]['r2'].append(r2)
                                    # Store results for each building
                                    building_key = (building_number, direction, data_type)
                                    if building_key not in building_results:
                                        building_results[building_key] = {'mse': [], 'mae': [], 'r2': []}

                                    building_results[building_key]['mse'].append(mse)
                                    building_results[building_key]['mae'].append(mae)
                                    building_results[building_key]['r2'].append(r2)

                    else:
                        print(f'{target_dir} does not exist')

    # Print aggregated results
    for key, metrics in results.items():
        building, direction, floor, data_type = key
        avg_mse = np.mean(metrics['mse'])
        avg_mae = np.mean(metrics['mae'])
        avg_r2 = np.mean(metrics['r2'])
        # print(f'Building: {building}, Direction: {direction}, Floor: {floor}, Type: {data_type}')
        # print(f'Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}, Average R2: {avg_r2:.4f}\n')

    # Print aggregated results for each building
    for key, metrics in building_results.items():
        building, direction, data_type = key
        avg_mse = np.mean(metrics['mse'])
        avg_mae = np.mean(metrics['mae'])
        avg_r2 = np.mean(metrics['r2'])
        # print(f'Building: {building}, Direction: {direction}, Type: {data_type}')
        # print(f'Overall Average MSE: {avg_mse:.4f}, Overall Average MAE: {avg_mae:.4f}, Overall Average R2: {avg_r2:.4f}\n')
        print(f'{avg_mse:.4f}, {avg_mae:.4f}, {avg_r2:.4f}')

# Base directory where the 'pretrained_model/visualization' folder is located
# base_dir = 'pretrained_model/visualization'
base_dir = 'D:/visualization_lora'

building_numbers = None
floors_numbers = None
waves = ('wave5')
# floors_numbers = ['6', '7', '6', '7', '7', '4', '2', '2', '5', '5', '10', '4', '3', '5', '5', '8', '7']
# building_numbers = ['03', '21', '24', '27', '34', '40', '45', '47', '60', '61', '70-2', '71', '73', '75', '82', '86-3', '8990']

directions = ['x']
data_types = ['U']
# Call with both custom direction and data type
read_text_files(base_dir, building_numbers, floors_numbers, waves, directions, data_types)