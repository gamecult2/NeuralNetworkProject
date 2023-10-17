import numpy as np
import matplotlib.pyplot as plt
from Units import *

def generate_cyclic_load(duration=10, sampling_rate=50, max_displacement=65):
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return t, cyclic_load

# Define the range of maximum displacements
min_max_displacement = np.arange(20 * mm, 75 * mm, 10 * mm)

# Create a single figure for all the subplots
fig, ax = plt.subplots(figsize=(10, 4))

# Plot all cyclic loads in the same window (subplots)
for max_disp in min_max_displacement:
    t, cyclic_load = generate_cyclic_load(max_displacement=max_disp)
    ax.plot(t, cyclic_load, label=f'Max Displacement = {max_disp}')

# Customize the plot
ax.set_xlabel('Time (s)')
ax.set_ylabel('Load')
ax.set_title('Cyclic Load vs. Time')
ax.grid(True)
ax.legend()

# Show the plot
plt.show()
