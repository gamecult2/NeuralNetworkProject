import numpy as np
import matplotlib.pyplot as plt
from Units import *


def generate_cyclic_load(duration=10, sampling_rate=100, max_displacement=65, frequency=1, displacement_slope=1):
    """Generates a cyclic load with a specified frequency, total length, increasing rate, and maximum displacement.

    Args:
        duration (float): The total length of the cyclic load in seconds.
        sampling_rate (int): The sampling rate of the cyclic load in Hz.
        max_displacement (float): The maximum displacement of the cyclic load.
        frequency (float): The frequency of the cyclic load in Hz.
        displacement_slope (float): The displacement slope of the cyclic load, which controls the increasing rate of the cyclic load.

    Returns:
        t (numpy.ndarray): A NumPy array containing the time steps of the cyclic load.
        cyclic_load (numpy.ndarray): A NumPy array containing the cyclic load values.
    """

    # Calculate the angular frequency
    angular_frequency = 2 * np.pi * frequency

    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = displacement_slope * t * np.sin(angular_frequency * t)

    return t, cyclic_load


# Define the range of maximum displacements
min_max_displacement = np.arange(75 * mm, 76 * mm, 1 * mm)
min_max_duration = np.arange(2, 10, 2)

# Create a single figure for all the subplots
fig, ax = plt.subplots(figsize=(10, 4))

# Plot all cyclic loads in the same window (subplots)
for max_disp in min_max_displacement:
    for max_duration in min_max_duration:
        t, cyclic_load = generate_cyclic_load(displacement_slope=max_duration, max_displacement=max_disp)
        ax.plot(t, cyclic_load, label=f'Max Displacement = {max_disp}')

# Customize the plot
ax.set_xlabel('Time (s)')
ax.set_ylabel('Load')
ax.set_title('Cyclic Load vs. Time')
ax.grid(True)
ax.legend()

# Show the plot
plt.show()
