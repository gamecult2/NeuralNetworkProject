import numpy as np
from Units import *
import random
import csv
import RCWall_Cyclic_Model as rcmodel

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
    0.01           # (loadcoef) axial load ratio
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
    0.1            # (loadcoef) axial load ratio
]

minDisplacement = [
    20 * mm
]
maxDisplacement = [
    75 * mm
]

# (FOR VALIDATION ONLY) Define the parameter ranges
# maxParameters = minParameters = [
#     102 * mm,      # (tw) Minimum thickness
#     3.66 * m,      # (hw) wall height
#     1.22 * m,      # (lw) wall length (min t*6)
#     190 * mm,      # (lbe) BE length (as a percentage of wall length)
#     41.7 * MPa,    # (fc) Concrete Compressive Strength
#     414 * MPa,     # (fy) Steel Yield Strength
#     0.03,          # (rhoBE) BE long reinforcement ratio (Minimum = 0.01)
#     0.0030,        # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
#     0.1]           # (loadcoef) axial load ratio
# minDisplacement = maxDisplacement = [
#     75 * mm]       # (loadcoef) Maximum displacement

# Set a seed to make random numbers reproducible
random.seed(22)

# Define the number of samples and the number of cyclic loads for each sample
num_samples = 50

# Open the CSV file for writing
with open("RCWall_Data/generated_samples.csv", 'a', newline='') as file:
    writer = csv.writer(file)

    for sample_index in range(num_samples):
        print("RUNNING SIMULATION N°", sample_index)

        # Generate parameter values for each sample
        tw = random.uniform(minParameters[0], maxParameters[0])
        hw = random.uniform(minParameters[1], maxParameters[1])
        lw = random.uniform(tw * 6, maxParameters[2])
        lbe = random.uniform(minParameters[3], maxParameters[3])
        fc = random.uniform(minParameters[4], maxParameters[4])
        fy = random.uniform(minParameters[5], maxParameters[5])
        rouYb = random.uniform(minParameters[6], maxParameters[6])
        rouYw = random.uniform(minParameters[7], maxParameters[7])
        loadcoef = random.uniform(minParameters[8], maxParameters[8])

        Max_Displacement = int(random.uniform(minDisplacement[0], maxDisplacement[0]))
        DisplacementStep = list(generate_cyclic_load(Max_Displacement))

        parameter_values = [tw, hw, lw, lbe, fc, fy, rouYb, rouYw, loadcoef]
        displacement_values = DisplacementStep

        print("USED PARAMETERS : ", parameter_values)

        rcmodel.build_model(tw, hw, lw, lbe, fc, fy, rouYb, rouYw, loadcoef)
        [x, y] = rcmodel.run_analysis(DisplacementStep, plotPushOverResults=False, printProgression=False)
        rcmodel.reset_analysis()
        print(len(x))

        if len(x) == 500:
            # Save all samples in the same CSV file
            # writer.writerow([sample_index])
            writer.writerow(['parameter_values'] + parameter_values)  # Start the line with headers
            writer.writerow(['displacement_values'] + displacement_values)
            writer.writerow(['y_values'] + y.astype(str).tolist())  # Convert 'y' to a list of strings
            # writer.writerow(x)


