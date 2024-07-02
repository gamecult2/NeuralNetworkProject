import os
import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


def calculate_metrics(target, output):
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    r2 = r2_score(target, output)
    R, p = pearsonr(target, output)
    return mse, mae, R


def plotting(gm_data, resp_data, title, save_fig=False):
    plt.rcParams.update({'font.size': 9, 'font.family': 'Cambria'})
    x_values = np.arange(0, len(gm_data))
    plt.figure(figsize=(5*0.75, 1.5))
    # plt.plot(x_values, gm_data, color='blue', linewidth=0.3, label='')
    plt.plot(x_values, resp_data, color='blue', linewidth=0.3, label='')
    ax = plt.gca()   # Get current axes

    plt.text(0.99, 0.03, f"Structural Response", transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
    # plt.text(0.99, 0.03, f"Ground Motion", transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.1)
    plt.xlim(xmin=0, xmax=max(x_values))
    plt.yticks(fontname='Cambria', fontsize=19)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    cmap = LinearSegmentedColormap.from_list("gradient", ["lightblue", "lightblue"])
    norm = plt.Normalize(vmin=0, vmax=200)

    ylim = ax.get_ylim()
    # Create the green rectangle patch from x = 0 to 150
    green_rect = patches.Rectangle((0, ylim[0]), 350, ylim[1] - ylim[0],
                                   linewidth=0, edgecolor=None, facecolor='blue', alpha=0.3)
    # ax.add_patch(green_rect)

    # Create the blue rectangle patch from x = 150 to 170
    blue_rect = patches.Rectangle((350, ylim[0]), 100, ylim[1] - ylim[0],
                                  linewidth=0, edgecolor=None, facecolor='green', alpha=0.3)
    # ax.add_patch(blue_rect)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'C:/Users/djerr/OneDrive/Bureau/Nature/Pictures/GMA/GMA.svg', format='svg', dpi=300, bbox_inches='tight')
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
            for floor_number in range(3, int(floor_count) + 1):  # Loop up to floor_count (inclusive)
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
                                    target_data = np.array([float(x) for x in lines[1].split()])
                                    gm_data = np.array([float(x) for x in lines[2].split()])
                                    title = (f'Direction_{direction.capitalize()}, GMA_{wave_name}')
                                    print(title)
                                    plotting(gm_data, target_data, title, save_fig=True)

                    else:
                        print(f'{target_dir} does not exist')


# Base directory where the 'pretrained_model/visualization' folder is located

base_dir = 'D:\pretrained_model\/visualization'

building_numbers = ['03', '21', '24', '27', '34', '40', '45', '47', '60', '61', '70-2', '71', '73', '75', '82', '86-3', '8990']
floors_numbers =   [ '6',  '7',  '6',  '7',  '7',  '4',  '2',  '2',  '5',  '5',   '10',  '4',  '3',  '5',  '5',   '8' ,     '7']

directions = ['x']
data_types = ['A', 'U']

read_text_files(base_dir, building_numbers, floors_numbers, directions, data_types)
