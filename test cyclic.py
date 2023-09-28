import numpy as np
import matplotlib.pyplot as plt

def generate_cyclic_loading_history(duration_per_amplitude, max_displacement, num_amplitudes=12, num_per_amplitudes=2):
    # Calculate the time step based on the duration_per_amplitude
    dt = 0.01  # Your desired constant time step

    # Calculate the number of time steps per amplitude duration
    steps_per_duration = int(duration_per_amplitude / dt)

    # Calculate the step size between amplitudes
    step_size = max_displacement / num_amplitudes

    # Initialize an empty array to store the loading history
    loading_history = []

    # Generate the loading history with step-wise increasing displacements
    for i in range(num_amplitudes):
        displacement = (i + 1) * step_size  # Increase the displacement step-wise

        # Generate two cycles at each displacement
        for _ in range(num_per_amplitudes):
            t_one_cycle = np.linspace(0, duration_per_amplitude, num=steps_per_duration)
            loading_history = np.concatenate((loading_history, displacement * np.sin(2 * np.pi * t_one_cycle)))

    # Create a time array for the entire loading history
    t_total = np.linspace(0, len(loading_history) * dt, num=len(loading_history))

    # Plot the loading history
    plt.figure(figsize=(10, 4))
    plt.plot(t_total, loading_history)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Cyclic Deformation-Controlled Loading History with Step-wise Increasing Displacements (Constant dt)")
    plt.grid(True)
    plt.show()

# Define parameters for the cyclic loading history
duration_per_amplitude = 1  # Duration of each amplitude level in seconds
max_displacement = 20  # Maximum displacement in meters

# Call the function with the specified parameters
generate_cyclic_loading_history(duration_per_amplitude, max_displacement)
