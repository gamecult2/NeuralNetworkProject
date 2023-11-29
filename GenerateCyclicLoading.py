import numpy as np
import matplotlib.pyplot as plt


def generate_increasing_cyclic_loading(num_cycles=6, initial_displacement=0, max_displacement=60, num_points=50, repetition_cycles=1):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # amplitude = initial_displacement + max_displacement_increase * i / num_cycles
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


displacement = generate_increasing_cyclic_loading(num_cycles=5, initial_displacement=10, max_displacement=54, num_points=50, repetition_cycles=2)

# Plot the cyclic loading
print(len(displacement))
plt.plot(displacement, linewidth=1, marker='o', markersize=2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title(f'Cyclic Loading with Initial Displacement and Increasing Amplitude')
plt.grid(True)
plt.show()

