import numpy as np
import matplotlib.pyplot as plt


def generate_increasing_cyclic_loading_1000(num_cycles, initial_displacement, max_displacement, repetition_cycles):
    target_displacement_step_length = 1000  # Desired displacement step length

    # Calculate the required number of points
    num_points = target_displacement_step_length // (num_cycles * repetition_cycles)

    # Generate the time array
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)

    # Initialize the displacement array
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # Calculate the amplitude for the current cycle
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)

        # Generate the displacement for the current cycle
        displacement_segment = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

        # Add the displacement segment to the overall displacement array
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = displacement_segment

    return time, displacement


def generate_increasing_cyclic_loading(num_cycles=10, initial_displacement=0, max_displacement=60, num_points=50, repetition_cycles=1):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # amplitude = initial_displacement + max_displacement_increase * i / num_cycles
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return time, displacement


def generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, frequency=1, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = max_displacement * (i + 1) / num_cycles
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * frequency * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return time, displacement


def generate_increasing_cyclic_loading_with_final_amplitude(num_cycles, initial_displacement, max_displacement, frequency=1, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * frequency * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return time, displacement


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

# Example usage
num_cycles = 10
initial_displacement = 20
max_displacement = 50
repetition_cycles = 2
time1, displacement1 = generate_increasing_cyclic_loading_1000(num_cycles, initial_displacement, max_displacement, repetition_cycles)

# time1, displacement1 = generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, frequency=1, num_points=50, repetition_cycles=repetition_cycles)
# time2, displacement2 = generate_increasing_cyclic_loading_with_final_amplitude(num_cycles, initial_displacement, max_displacement+1, frequency=1, num_points=50, repetition_cycles=repetition_cycles)
# time3, displacement3 = generate_increasing_cyclic_loading_with_exponential_growth(num_cycles, initial_displacement, max_displacement+2, frequency=1, num_points=50, repetition_cycles=repetition_cycles)
# time4, displacement4 = generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points=50, repetition_cycles=repetition_cycles)

print(len(displacement1))

# Plot the cyclic loading
plt.plot(time1, displacement1)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title(f'Cyclic Loading with Initial Displacement and Exponential Growth in Amplitude (Last Cycle Reaches Final Amplitude)')
plt.grid(True)
plt.show()
