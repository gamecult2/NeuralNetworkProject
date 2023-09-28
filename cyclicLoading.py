import numpy as np
import matplotlib.pyplot as plt


def generate_cyclic_load(duration=12.0, sampling_rate=100, max_displacement=60):
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return t, cyclic_load


# Call the function to generate the cyclic load with the desired max_displacement
t, cyclic_load = generate_cyclic_load()

# Plot the cyclic load
plt.figure(figsize=(10, 4))
plt.plot(t, cyclic_load)
plt.xlabel('Time (s)')
plt.ylabel('Load')
plt.title('Cyclic Load with Maximum Displacement of 20')
plt.grid(True)
plt.show()
