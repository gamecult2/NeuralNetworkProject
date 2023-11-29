import numpy as np

def generate_linearly_increasing_cyclic_curve(periods=10, max_displacement=75, sampling_rate=50):
    duration = periods * 2 * np.pi  # Duration for the specified number of periods
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Calculate the linearly increasing amplitude for each period
    amplitude = np.linspace(0, max_displacement, num=num_samples // 2)  # Divide by 2 for two cycles per period
    amplitude = np.tile(amplitude, 2)  # Repeat the amplitude pattern for two cycles

    # Generate the cyclic curve with linearly increasing amplitude
    cyclic_curve = amplitude * np.sin(2 * np.pi * t)

    return cyclic_curve

# Example usage
cyclic_curve = generate_linearly_increasing_cyclic_curve(periods=10, max_displacement=75, sampling_rate=50)


def generate_cyclic_loading_history(duration_per_amplitude, max_displacement):
    # Set the number of amplitudes to 12
    num_amplitudes = 10
    num_per_amplitudes = 2

    # Create a time array for one cycle
    t_one_cycle = np.linspace(0, duration_per_amplitude, num=100)

    # Initialize an empty array to store the loading history
    loading_history = []

    # Generate the loading history with exponentially increasing displacements
    for i in range(num_amplitudes):
        # Calculate displacement exponentially
        displacement = max_displacement * (1.4 ** i)

        # Generate two cycles at each displacement
        for _ in range(num_per_amplitudes):
            loading_history = np.concatenate((loading_history, displacement * np.sin(2 * np.pi * t_one_cycle)))

    # Create a time array for the entire loading history with two decimal places
    t_total = np.round(np.linspace(0, duration_per_amplitude * num_amplitudes * num_per_amplitudes, num=len(loading_history)), 2)

    return t_total, loading_history