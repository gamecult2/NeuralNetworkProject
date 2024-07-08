import numpy as np
from sklearn.decomposition import PCA

# Set the parameters
batch_size = 4  # Increase the batch size to have more samples
time_steps = 10
num_parameters = 10

# Example parameters for three samples
parameters = np.array([
    [102, 3810, 1220, 190, 42, 434, 448, 0.029, 0.003, 0.092],
    [102, 3810, 1220, 190, 42, 434, 448, 0.029, 0.003, 0.092],
    [102, 3810, 1220, 190, 42, 434, 448, 0.029, 0.003, 0.0921],
    [102, 3810, 1220, 190, 42, 434, 448, 0.029, 0.00301, 0.092]
])

# Generate displacement data uniformly from 1 to 10 for each sample in the batch
displacement = np.array([np.linspace(1, 10, time_steps) for _ in range(batch_size)]).reshape(batch_size, time_steps, 1)

# Reduce parameters using PCA to 2 components
pca = PCA(n_components=2)
reduced_parameters = pca.fit_transform(parameters)
reduced_parameters = reduced_parameters.reshape(batch_size, 1, 2)

# Concatenate the reduced parameters with the displacement
reduced_parameters_repeated = np.repeat(reduced_parameters, time_steps, axis=1)
combined = np.concatenate([displacement, reduced_parameters_repeated], axis=-1)

# Print the results
print("Parameters before PCA:\n", parameters)
print("\nReduced parameters after PCA:\n", reduced_parameters)
print("\nDisplacement data:\n", displacement)
print("\nCombined tensor shape:", combined.shape)
print("\nCombined tensor (first 10 lines of the first sample):\n", combined[0, :10, :])