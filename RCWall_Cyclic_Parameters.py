import numpy as np
from Units import *
import random
import csv


def generate_cyclic_load(duration=10, sampling_rate=50, max_displacement=65):
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return t, cyclic_load

# Define the parameter ranges
minParameters = [
    150 * mm,      # (tw) Minimum thickness
    1 * m,         # (hw) wall height
    None,          # (lw) wall length (min t*6)
    100 * mm,      # (lbe) BE length (as a percentage of wall length)
    20 * MPa,      # (fc) Concrete Compressive Strength
    380 * MPa,     # (fy) Steel Yield Strength
    0.01,          # () BE long reinf ratio (Minimum = 0.01)
    0.0025,        # () WEB long reinf ratio (Minimum = 0.0025)
    0.01           # () axial load ratio
]

maxParameters = [
    400 * mm,      # (tw) Maximum thickness
    6 * m,         # (hw) wall height
    3 * m,         # (lw) wall length (min t*6)
    300 * mm,      # (lbe) BE length (as a percentage of wall length)
    70 * MPa,      # (fc) Concrete Compressive Strength
    630 * MPa,     # (fy) Steel Yield Strength
    0.04,          # (rhoBE) BE long reinf ratio
    0.025,         # (rhoWEB) WEB long reinf ratio
    0.1           # () axial load ratio
]

# Set a seed to make random numbers reproducible
random.seed(123)  # You can use any integer as the seed

# Define the number of samples
num_samples = 3
num_cyclic = 4

# Create a list to store all the samples
all_samples = []

# Generate arrays of random values for each parameter using a single loop
for sample_index in range(num_samples):
    tw = random.uniform(minParameters[0], maxParameters[0])
    hw = random.uniform(minParameters[1], maxParameters[1])
    lw = random.uniform(tw * 6, maxParameters[2])
    lbe = random.uniform(minParameters[3], maxParameters[3])
    rouYb = random.uniform(minParameters[4], maxParameters[4])
    rouYw = random.uniform(minParameters[5], maxParameters[5])
    loadcoef = random.uniform(minParameters[6], maxParameters[6])
    fc = random.uniform(minParameters[7], maxParameters[7])
    fy = random.uniform(minParameters[8], maxParameters[8])

    # Create a list of the parameter values for this sample
    parameter_values = [tw, hw, lw, lbe, rouYb, rouYw, loadcoef, fc, fy]

    all_samples.append(parameter_values)

    # Define the range of maximum displacements
    min_max_displacement = np.arange(10, 121, 10)
    # Plot all cyclic loads in the same window (subplots)
    for max_disp in min_max_displacement:
        t, cyclic_load = generate_cyclic_load(max_displacement=max_disp)

    all_samples.append(cyclic_load)


    # Save all samples in the same CSV file
with open("generated_samples.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["tw", "hw", "lw", "lbe", "rouYb", "rouYw", "loadcoef", "fc", "fy"])
    writer.writerows(all_samples)




