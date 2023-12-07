import numpy as np
from Units import *
import random
import csv

random.seed(22)

def generate_cyclic_load(max_displacement=65):
    duration = 10
    sampling_rate = 50
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return cyclic_load

# Define the parameter ranges
minParameters = [
    150 * mm,      # (tw) Minimum thickness
    1 * m,         # (hw) wall height
    None,          # (lw) wall length (min t*6)
    100 * mm,      # (lbe) BE length (as a percentage of wall length)
    20 * MPa,      # (fc) Concrete Compressive Strength
    380 * MPa,     # (fy) Steel Yield Strength
    0.01,          # (rhoBE) BE long reinforcement ratio (Minimum = 0.01)
    0.0025,        # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
    0.01           # (loadCoeff) axial load ratio
]

maxParameters = [
    400 * mm,      # (tw) Maximum thickness
    6 * m,         # (hw) wall height
    3 * m,         # (lw) wall length (min t*6)
    300 * mm,      # (lbe) BE length (as a percentage of wall length)
    70 * MPa,      # (fc) Concrete Compressive Strength
    630 * MPa,     # (fy) Steel Yield Strength
    0.04,          # (rhoBE) BE long reinforcement ratio
    0.025,         # (rhoWEB) WEB long reinforcement ratio
    0.1            # (loadCoeff) axial load ratio
]

minDisplacement = [
    10 * mm
]

maxDisplacement = [
    200 * mm
]

# Define the number of samples and the number of cyclic loads for each sample
num_samples = 1
num_cyclic = 3

# Create a list to store all the samples
all_samples = []

parameter_labels = ["tw", "hw", "lw", "lbe", "fc", "fy", "rouYb", "rouYw", "loadCoeff"] + ["cyclic_displacement"]

for sample_index in range(num_samples):
    # Generate parameter values for the main sample
    tw = random.uniform(minParameters[0], maxParameters[0])
    hw = random.uniform(minParameters[1], maxParameters[1])
    lw = random.uniform(tw * 6, maxParameters[2])
    lbe = random.uniform(minParameters[3], maxParameters[3])
    fc = random.uniform(minParameters[4], maxParameters[4])
    fy = random.uniform(minParameters[5], maxParameters[5])
    rouYb = random.uniform(minParameters[6], maxParameters[6])
    rouYw = random.uniform(minParameters[7], maxParameters[7])
    loadcoef = random.uniform(minParameters[8], maxParameters[8])

    # Generate additional samples for the main sample
    for cyclic_index in range(1, num_cyclic + 1):
        Max_Displacement = int(random.uniform(minDisplacement[0], maxDisplacement[0]))
        cyclic_displacement = list(generate_cyclic_load(Max_Displacement))

        # Create a list of the parameter values for this sample
        parameter_values = [tw, hw, lw, lbe, fc, fy, rouYb, rouYw, loadcoef] + cyclic_displacement

        # Append the main sample to the list
        all_samples.append(parameter_values)

# Save all samples in the same CSV file
with open("generated_samples.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(parameter_labels)
    for row in all_samples:
        writer.writerow(row)

