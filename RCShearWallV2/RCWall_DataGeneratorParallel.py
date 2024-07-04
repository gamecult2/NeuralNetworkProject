import csv
import random
import math
import numpy as np
import os
import sys
from multiprocessing import Pool, cpu_count


def generate_sample(args):
    worker_id, sample_index = args

    # Set up isolated environment
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Import modules here to ensure they're loaded in the isolated environment
    import RCWall_Model_SFI as rcmodel
    from RCWall_ParametersRange import minParameters, maxParameters, minLoading, maxLoading
    from GenerateCyclicLoading import generate_increasing_cyclic_loading_with_repetition

    # Set a unique random seed for this worker
    random.seed(worker_id * 1000000 + sample_index)
    np.random.seed(worker_id * 1000000 + sample_index)

    # Generate parameters
    print(f"========================================= RUNNING SAMPLE \033[92mN° {sample_index}\033[0m =========================================")
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
        # Write to worker-specific CSV file
        with open(f"RCWall_Data/RCWall_Dataset_Worker_{worker_id}.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(parameter_values)
            writer.writerow(DisplacementStep[:-1])
            writer.writerow(np.concatenate((y1[:1], y1[2:])))
        return 1  # Return 1 for valid sample
    return 0  # Return 0 for invalid sample


def init_worker(id):
    global worker_id, sequence_length
    worker_id = id
    sequence_length = 501


def process_chunk(args):
    worker_id, chunk = args
    valid_samples = sum(generate_sample((worker_id, i)) for i in chunk)
    return valid_samples


if __name__ == "__main__":
    num_samples = 1000000
    chunk_size = 1000
    num_processes = cpu_count()

    # Create output directory if it doesn't exist
    os.makedirs("RCWall_Data", exist_ok=True)

    chunks = [range(i, min(i + chunk_size, num_samples)) for i in range(0, num_samples, chunk_size)]

    with Pool(processes=num_processes, initializer=init_worker, initargs=(range(num_processes),)) as pool:
        results = pool.imap_unordered(process_chunk, zip(range(num_processes), chunks))

        total_valid_samples = 0
        for i, valid_samples in enumerate(results):
            total_valid_samples += valid_samples
            print(f"Processed chunk {i + 1}/{len(chunks)}. Valid samples in this chunk: {valid_samples}")

    print(f"Data generation complete. Total valid samples: {total_valid_samples}")