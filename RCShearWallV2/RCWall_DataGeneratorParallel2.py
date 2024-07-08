import csv
import random
import math
import numpy as np
import os
from multiprocessing import Pool, cpu_count, current_process

# Import RCWall_Model and functions
import RCWall_Model_SFI as rcmodel
from RCWall_ParametersRange import minParameters, maxParameters, minLoading, maxLoading
from GenerateCyclicLoading import generate_increasing_cyclic_loading_with_repetition

# Global constants
sequence_length = 501

def generate_sample(sample_index):
    # Set a unique random seed for each sample
    random_seed = random.randint(0, 2**32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)
    worker_id = current_process().name

    print(f"===================================== RUNNING SAMPLE \033[92mN° {sample_index}\033[0m  in Core \033[92mN°{worker_id}\033[0m =====================================\n")

    # Generate parameters
    tw = round(random.uniform(minParameters[0], maxParameters[0]))
    tb = round(random.uniform(tw, maxParameters[1]))
    hw = round(random.uniform(minParameters[2], maxParameters[2]) / 10) * 10
    lw = round(random.uniform(tw * 6, maxParameters[3]) / 10) * 10
    lbe = round(random.uniform(lw * minParameters[4], lw * maxParameters[4]))
    fc = round(random.uniform(minParameters[5], maxParameters[5]))
    fyb = round(random.uniform(minParameters[6], maxParameters[6]))
    fyw = round(random.uniform(minParameters[7], maxParameters[7]))
    rouYb = round(random.uniform(minParameters[8], maxParameters[8]), 4)
    rouYw = round(random.uniform(minParameters[9], maxParameters[9]), 4)
    rouXb = round(random.uniform(minParameters[10], maxParameters[10]), 4)
    rouXw = round(random.uniform(minParameters[11], maxParameters[11]), 4)
    loadCoeff = round(random.uniform(minParameters[12], maxParameters[12]), 4)

    num_cycles = int(random.uniform(minLoading[0], maxLoading[0]))
    max_displacement = int(random.uniform(hw * 0.005, hw * 0.040))
    repetition_cycles = int(random.uniform(minLoading[2], maxLoading[2]))
    num_points = math.ceil(sequence_length / (num_cycles * repetition_cycles))

    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points, repetition_cycles)[:sequence_length]

    parameter_values = [tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadCoeff]

    # Run analysis
    rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadCoeff, printProgression=False)
    rcmodel.run_gravity(printProgression=False)
    y1 = rcmodel.run_cyclic2(DisplacementStep, plotResults=False, printProgression=False, recordData=False)
    rcmodel.reset_analysis()

    if len(y1) == sequence_length:
        return worker_id, parameter_values, DisplacementStep[:-1], np.concatenate((y1[:1], y1[2:]))

    return None

def write_sample_data(worker_id, data):
    os.makedirs("RCWall_Data", exist_ok=True)
    with open(f"RCWall_Data/Worker_{worker_id}_Data.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[1])
        writer.writerow(data[2])
        writer.writerow(data[3])

def process_chunk(chunk):
    valid_samples = []
    for sample_index in chunk:
        sample_result = generate_sample(sample_index)
        if sample_result is not None:
            valid_samples.append(sample_result)
            write_sample_data(sample_result[0], sample_result)
    return len(valid_samples)

if __name__ == "__main__":
    num_samples = 160
    num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        chunks = np.array_split(range(num_samples), num_processes)
        results = pool.map(process_chunk, chunks)

    total_valid_samples = sum(results)
    print(f"Data generation complete. Total valid samples generated: {total_valid_samples}")
