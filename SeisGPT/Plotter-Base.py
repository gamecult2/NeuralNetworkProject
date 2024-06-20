import os
import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def calculate_metrics(target, output):
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    r2 = r2_score(target, output)
    R, p = pearsonr(target, output)
    return mse, mae, R


def plotting(target, predicted, x_label, data_type, title, mse, mae, r, save_fig=False):

    plt.rcParams.update({'font.size': 9, "font.family": ["Cambria", "Cambria"]})

    x_values = np.arange(0, len(target) * 0.02, 0.02)[::4]
    plt.figure(figsize=(6*0.75, 3*0.75))
    plt.plot(x_values, target[::4], color='blue', linewidth=0.3, label='FEA')
    plt.plot(x_values, predicted[::4], color='red', linewidth=0.15, label='SeisGPT-Base')  # Include MSE with line break
    ax = plt.gca()   # Get current axes
    plt.text(0.99, 0.03, f"MSE: {mse:.3f}, MAE: {mae:.3f}, R: {r:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'size': 9}
    if data_type == "A":
        y_label = 'Acceleration (m/s\xb2)'
    elif data_type == "U":
        y_label = 'Displacement (m)'
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.xlim(xmin=0, xmax=max(x_values*1))
    plt.yticks(fontname='Cambria', fontsize=9)
    plt.xticks(fontname='Cambria', fontsize=9)
    plt.tick_params(axis='both', which='both', direction='in')  # Adjust padding if needed
    plt.title(f"{title}", fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 9})
    plt.tight_layout()
    plt.subplots_adjust(left=0.101, bottom=0.203, right=0.985, top=0.888, hspace=0.23, wspace=0.2)  # Adjust margins and spacing
    plt.legend(prop={'size': 9}, frameon=False, loc='upper right')  # Adjust slightly for better placement
    if save_fig:
        building_name = title.split(", ")[0]
        folder_path = f'C:/Users/djerr/OneDrive/Bureau/Nature/Pictures/Validation/Pre-trained/{building_name}/{data_type}/'  # Pre-trained/
        print(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(f'{folder_path}/{title}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()


def read_text_files(base_dir, building_numbers=None, floors_numbers=None, directions=None, data_types=None):
    # Allow for optional building number list
    if not floors_numbers:
        floors_numbers = ['6', '7', '6', '7', '7', '4', '2', '2', '5', '5', '10', '4', '3', '5', '5', '8', '7']
    if not building_numbers:
        building_numbers = ['03', '21', '24', '27', '34', '40', '45', '47', '60', '61', '70-2', '71', '73', '75', '82', '86-3', '8990']
    if not directions:
        directions = ['x', 'y']
    if not data_types:
        data_types = ['A', 'U']

    # Loop through each building
    for building_number, floor_count in zip(building_numbers, floors_numbers):
        for direction in directions:
            building_dir = f'tongji_{building_number}_elastic_{direction.lower()}'
            # Loop through each floor
            # for floor_number in floor_numbers:
            for floor_number in range(2, int(floor_count) + 1):  # Loop up to floor_count (inclusive)
                floor_dir = f'layer_{floor_number}'
                # Loop through Displacement (A) and Acceleration (U) folders
                for data_type in data_types:
                    data_type = f'{data_type}'
                    target_dir = os.path.join(base_dir, building_dir, floor_dir, data_type)
                    print(target_dir)
                    # Check if directory exists
                    if os.path.exists(target_dir):
                        txt_files = glob.glob(os.path.join(target_dir, '*.txt'))
                        for txt_file in txt_files:
                            with open(txt_file, 'r') as file:
                                lines = file.readlines()
                                wave_name = os.path.splitext(os.path.basename(txt_file))[0]
                                if len(lines) >= 3:
                                    target_data = np.array([float(x) for x in lines[0].split()])
                                    output_data = np.array([float(x) for x in lines[1].split()])
                                    gm_data = np.array([float(x) for x in lines[2].split()])
                                    mse, mae, r = calculate_metrics(target_data, output_data)
                                    title = (f'Building_{building_number}, Floor_{floor_number}, Direction_{direction.capitalize()}, {wave_name}')
                                    print(title)
                                    plotting(target_data, output_data, 'Time (s)', data_type, title, mse, mae, r, save_fig=False)

                    else:
                        print(f'{target_dir} does not exist')


# Base directory where the 'pretrained_model/visualization' folder is located

base_dir = 'pretrained_model/visualization'

# building_numbers = ['03', '21', '24', '27', '34', '40', '45', '47', '60', '61', '70-2', '71', '73', '75', '82', '86-3', '8990']
# floors_numbers =   [ '6',  '7',  '6',  '7',  '7',  '4',  '2',  '2',  '5',  '5',   '10',  '4',  '3',  '5',  '5',   '8' ,     '7']
building_numbers = ['70-2']
floors_numbers   = ['10']

directions = ['x']
data_types = ['A', 'U']

read_text_files(base_dir, building_numbers, floors_numbers, directions, data_types)
