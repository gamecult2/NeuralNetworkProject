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

def plotting(target, wave1, wave3, wave5, x_label, data_type, title, mse, mae, r, save_fig=False):

    plt.rcParams.update({'font.size': 9, "font.family": ["Cambria", "Cambria"]})
    skip = 4
    x_values = np.arange(0, len(target) * 0.02, 0.02)[::skip]
    plt.figure(figsize=(4.5*2, 2.25))
    plt.plot(x_values, target[::skip], color='blue', linestyle='-', linewidth=0.3, label='FEA')
    plt.plot(x_values, wave1[::skip], color='red', linestyle='-', linewidth=0.2, label='SeisGPT-Enhanced (1 wave)')
    plt.plot(x_values, wave3[::skip], color='green', linestyle='-', linewidth=0.2, label='SeisGPT-Enhanced (3 waves)')
    plt.plot(x_values, wave5[::skip], color='black', linestyle='-', linewidth=0.2, label='SeisGPT-Enhanced (5 waves)')
    ax = plt.gca()
    # plt.text(0.99, 0.03, f"MSE: {mse:.3f}, MAE: {mae:.3f}, R: {r:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'size': 9}
    y_label = 'Acceleration (m/s²)' if data_type == "A" else 'Displacement (m)'
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.xlim(xmin=0, xmax=max(x_values*1))
    plt.yticks(fontname='Cambria', fontsize=9)
    plt.xticks(fontname='Cambria', fontsize=9)
    plt.tick_params(axis='both', which='both', direction='in')
    plt.title(f"{title}", fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 9})
    plt.tight_layout()
    plt.subplots_adjust(left=0.101, bottom=0.203, right=0.985, top=0.888, hspace=0.23, wspace=0.2)
    plt.legend(prop={'size': 9}, frameon=False, loc='upper right')
    if save_fig:
        building_name = title.split(", ")[0]
        folder_path = f'C:/Users/djerr/OneDrive/Bureau/Nature/Pictures/Validation/Fine-tuned 3/{building_name}/{data_type}/'
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(f'{folder_path}/{title}.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.show()

def read_text_files(base_dir, building_numbers=None, floors_numbers=None, directions=None, data_types=None):
    if not floors_numbers:
        floors_numbers = ['6', '7', '6', '7', '7', '4', '2', '2', '5', '5', '10', '4', '3', '5', '5', '8', '7']
    if not building_numbers:
        building_numbers = ['03', '21', '24', '27', '34', '40', '45', '47', '60', '61', '70-2', '71', '73', '75', '82', '86-3', '8990']
    if not directions:
        directions = ['x', 'y']
    if not data_types:
        data_types = ['A', 'U']

    for building_number, floor_count in zip(building_numbers, floors_numbers):
        for direction in directions:
            building_dir = f'tongji_{building_number}_elastic_{direction.lower()}'
            for floor_number in range(2, int(floor_count) + 1):
                floor_dir = f'layer_{floor_number}'
                for data_type in data_types:
                    data_type = f'{data_type}'
                    wave1_dir = os.path.join(base_dir, building_dir, floor_dir, 'wave1', data_type)
                    wave3_dir = os.path.join(base_dir, building_dir, floor_dir, 'wave3', data_type)
                    wave5_dir = os.path.join(base_dir, building_dir, floor_dir, 'wave5', data_type)
                    print(wave1_dir)
                    if os.path.exists(wave1_dir) and os.path.exists(wave3_dir) and os.path.exists(wave5_dir):
                        wave1_files = glob.glob(os.path.join(wave1_dir, '*.txt'))
                        wave3_files = glob.glob(os.path.join(wave3_dir, '*.txt'))
                        wave5_files = glob.glob(os.path.join(wave5_dir, '*.txt'))
                        for wave1_file, wave3_file, wave5_file in zip(wave1_files, wave3_files, wave5_files):
                            with open(wave1_file, 'r') as file1, open(wave3_file, 'r') as file3, open(wave5_file, 'r') as file5:
                                lines1 = file1.readlines()
                                lines3 = file3.readlines()
                                lines5 = file5.readlines()
                                wave_name = os.path.splitext(os.path.basename(wave1_file))[0]
                                if len(lines1) >= 3 and len(lines3) >= 3 and len(lines5) >= 3:
                                    target_data = np.array([float(x) for x in lines1[0].split()])
                                    wave1_data = np.array([float(x) for x in lines1[1].split()])
                                    wave3_data = np.array([float(x) for x in lines3[1].split()])
                                    wave5_data = np.array([float(x) for x in lines5[1].split()])
                                    mse, mae, r = calculate_metrics(target_data, wave1_data)
                                    title = (f'Building_{building_number}, Floor_{floor_number}, Direction_{direction.capitalize()}, {wave_name}')
                                    print(title)
                                    plotting(target_data, wave1_data, wave3_data, wave5_data, 'Time (s)', data_type, title, mse, mae, r, save_fig=True)
                    else:
                        print(f'{wave1_dir}, {wave3_dir}, or {wave5_dir} does not exist')

base_dir = 'D:/visualization_lora'

building_numbers = ['8990']
floors_numbers   = [   '7']
directions = ['x']
data_types = ['A', 'U']

read_text_files(base_dir, building_numbers, floors_numbers, directions, data_types)
