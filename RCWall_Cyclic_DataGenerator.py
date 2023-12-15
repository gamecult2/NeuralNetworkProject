import numpy as np
from Units import *
import random
import math
import csv
import RCWall_Cyclic_Model as rcmodel
import RCWall_Cyclic_Model_simple as rcmodel

random.seed(22)


def generate_cyclic_load(max_displacement=75):
    duration = 10
    sampling_rate = 50
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)
    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return cyclic_load


def generate_increasing_cyclic_loading(num_cycles=10, initial_displacement=5, max_displacement=60, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # amplitude = initial_displacement + max_displacement_increase * i / num_cycles
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


def generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = max_displacement * (i + 1) / num_cycles
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


def generate_increasing_cyclic_loading_with_exponential_growth(num_cycles, initial_displacement, max_displacement, frequency=1, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # Use exponential growth function for amplitude
        growth_factor = (max_displacement / initial_displacement) ** (1 / (num_cycles - 1))
        amplitude = initial_displacement * growth_factor ** i
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2 * np.pi * frequency * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    # Ensure the last cycle has the final amplitude
    displacement[-num_points * repetition_cycles:] = max_displacement * np.sin(2 * np.pi * frequency * time[-num_points * repetition_cycles:])

    return time, displacement


# ***************************************************************************************************
#           DEFINE PARAMETER RANGES
# ***************************************************************************************************
# Define the parameter ranges
minParameters = [
    150.0 * mm,  # (tw) Minimum thickness
    1.0 * m,  # (hw) wall height
    None,  # (lw) wall length (min t*6)
    100.0 * mm,  # (lbe) BE length (as a percentage of wall length)
    20.0 * MPa,  # (fc) Concrete Compressive Strength
    275.0 * MPa,  # (fyb) Steel Yield Strength BE
    275.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.0050,  # (rhoBE) BE long reinforcement ratio (Minimum = 0.01)
    0.0020,  # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
    0.0100  # (loadCoeff) axial load ratio
]

maxParameters = [
    400.0 * mm,  # (tw) Maximum thickness
    6.0 * m,  # (hw) wall height
    4.0 * m,  # (lw) wall length (min t*6)
    300.0 * mm,  # (lbe) BE length (as a percentage of wall length)
    70.0 * MPa,  # (fc) Concrete Compressive Strength
    650.0 * MPa,  # (fyb) Steel Yield Strength BE
    650.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.0550,  # (rhoBE) BE long reinforcement ratio
    0.0300,  # (rhoWEB) WEB long reinforcement ratio
    0.1000  # (loadCoeff) axial load ratio
]

# Define the parameter ranges
minLoading = [
    6,  # num_cycles
    # 1 * mm,  # initial_displacement
    20 * mm,  # max_displacement
    1  # repetition_cycles
]

maxLoading = [
    12,  # num_cycles
    # 10 * mm,  # initial_displacement
    200 * mm,  # max_displacement
    3  # repetition_cycles
]

# (FOR VALIDATION ONLY) Define the parameter ranges
# maxParameters = minParameters = [
#     102 * mm,      # (tw) Minimum thickness
#     3.81 * m,      # (hw) wall height
#     1.22 * m,      # (lw) wall length (min t*6)
#     190 * mm,      # (lbe) BE length (as a percentage of wall length)
#     41.7 * MPa,    # (fc) Concrete Compressive Strength
#     434 * MPa,     # (fyb) Steel Yield Strength BE
#     448 * MPa,     # (fyw) Steel Yield Strength Web
#     0.03,          # (rhoBE) BE long reinforcement ratio (Minimum = 0.01)
#     0.0030,        # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
#     0.0835]           # (loadCoeff) axial load ratio
#
# minLoading = maxLoading = [
#     10,  # num_cycles
#     5 * mm,  # initial_displacement
#     86 * mm,  # max_displacement
#     2]   # repetition_cycles]       # (loadCoeff) Maximum displacement

# ***************************************************************************************************
#           DEFINE NUMBER OF SAMPLE TO GENERATE
# ***************************************************************************************************
# Define the number of samples to be generated
num_samples = 10000
timeseries_length = 500

# Open the CSV file for writing
with open("RCWall_Data/RCWall_generated_samples(multiAnalysis).csv", 'a', newline='') as file:
    writer = csv.writer(file)
    converged = []
    nonconverged = []
    for sample_index in range(num_samples):
        print("RUNNING SIMULATION N°", sample_index)

        # ***************************************************************************************************
        #           GENERATE PARAMETERS FOR EACH SAMPLE
        # ***************************************************************************************************
        # Geometric parameters
        tw = random.uniform(minParameters[0], maxParameters[0])
        hw = random.uniform(minParameters[1], maxParameters[1])
        lw = random.uniform(tw * 6, maxParameters[2])
        lbe = random.uniform(minParameters[3], maxParameters[3])
        fc = random.uniform(minParameters[4], maxParameters[4])
        fyb = random.uniform(minParameters[5], maxParameters[5])
        fyw = random.uniform(minParameters[6], maxParameters[6])
        rouYb = random.uniform(minParameters[7], maxParameters[7])
        rouYw = random.uniform(minParameters[8], maxParameters[8])
        loadCoeff = random.uniform(minParameters[9], maxParameters[9])

        # Cyclic load parameters
        num_cycles = int(random.uniform(minLoading[0], maxLoading[0]))
        # initial_displacement = int(random.uniform(minLoading[1], maxLoading[1]))
        max_displacement = int(random.uniform(minLoading[1], maxLoading[1]))
        repetition_cycles = int(random.uniform(minLoading[2], maxLoading[2]))
        num_points = math.ceil(timeseries_length / (num_cycles * repetition_cycles))  # Ensure at least 1000 points in total.

        # DisplacementStep = list(generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles))
        DisplacementStep = list(generate_increasing_cyclic_loading_with_repetition(num_cycles,  max_displacement, num_points, repetition_cycles))
        DisplacementStep = DisplacementStep[: timeseries_length]  # Limit displacement of Cyclic analysis to 1000 points

        # Pushover parameters
        DispIncr = max_displacement / timeseries_length  # limit displacement for Pushover analysis to 1000 points

        # Overall parameters
        parameter_values = [round(value, 4) for value in [tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff]]
        cyclic_values = [round(value, 4) for value in [max_displacement, repetition_cycles, num_cycles, num_points]]
        pushover_values = [round(value, 4) for value in [max_displacement, DispIncr]]
        print("\033[92m USED PARAMETERS -> (Characteristic):", parameter_values, "-> (Cyclic Loading):", cyclic_values, "-> (Pushover Loading):", pushover_values, "\033[0m")

        # ***************************************************************************************************
        #           RUN ANALYSIS (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # CYCLIC ANALYSIS
        print("\033[92m Running Cyclic Analysis :", cyclic_values, "\033[0m", '--> DisplacementStep :', len(DisplacementStep),)
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        [x1, y1] = rcmodel.run_cyclic(DisplacementStep, plotResults=False, printProgression=False, recordData=False)
        rcmodel.reset_analysis()
#
        # # RUN PUSHOVER ANALYSIS
        print("\033[92m Running Pushover Analysis :", pushover_values, "\033[0m")
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        [x2, y2] = rcmodel.run_pushover(max_displacement, DispIncr, plotResults=False, printProgression=False, recordData=False)
        rcmodel.reset_analysis()

        # ***************************************************************************************************
        #           SAVE DATA (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # if 980 <= len(x1) <= 1020:
        # if 2980 <= len(x1) <= 3020 and 2980 <= len(x2) <= 3020:  # Check if the length of the response results is 1000 to write it to the file other results will be removed because of non-convergence
        if len(x1) == len(x2) == timeseries_length:
            converged.append(sample_index)
            # Save all samples in the same CSV file
            # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
            writer.writerow(['InputParameters_values'] + parameter_values)  # The 10 Parameters used for the simulation
            writer.writerow(['InputDisplacement_values'] + DisplacementStep)  # Cyclic Displacement imposed to the RC Shear Wall
            # ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
            writer.writerow(['OutputCyclicDisplacement_values'] + x1.astype(str).tolist())  # Displacement Response of the RC Shear Wall
            writer.writerow(['OutputCyclicShear_values'] + y1.astype(str).tolist())  # Shear Response of the RC Shear Wall
            # ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
            writer.writerow(['OutputPushoverDisplacement_values'] + x2.astype(str).tolist())  # Displacement Response of the RC Shear Wall
            writer.writerow(['OutputPushoverShear_values'] + y2.astype(str).tolist())  # Pushover Response of the RC Shear Wall
        else:
            nonconverged.append(sample_index)
        rcmodel.reset_analysis()

    # print('converged = ', len(converged))
    # print('Non-converged =', len(nonconverged))
