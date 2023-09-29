import numpy as np
import matplotlib.pyplot as plt
# Define the parameters
po_sequence_length = 100

# Generate linearly spaced displacement values
displacement = np.linspace(0, 10, po_sequence_length)

# Generate one curve resembling pushover analysis results in one line
curve_data = np.random.uniform(0.5, 2) * np.sin(np.random.uniform(0.1, 0.5) * displacement) * np.exp(-0.1 * (displacement - np.random.uniform(2, 8))**2)

plt.plot(displacement, curve_data, label=f'Mid Displacement: ')

plt.xlabel('Displacement')
plt.ylabel('Load')
plt.title('Curves Resembling Pushover Analysis Results')
plt.legend()
plt.grid(True)
plt.show()
