import numpy as np
import csv


def generate_peaks(Dmax, DincrStatic=0.01, CycleType="Full", Fact=1, Ncycles=1):

    # Scale Dmax.
    Dmax *= Fact

    # Calculate the total number of displacement increments.
    total_increments = Ncycles * 4 * int(abs(Dmax) / DincrStatic)

    # Create a vector of displacement increments.
    iDstep = np.zeros(total_increments, dtype=float)

    # Add the incremental displacements for each cycle.
    for cycle in range(Ncycles):
        start_index = cycle * 4 * int(abs(Dmax) / DincrStatic)
        end_index = start_index + 4 * int(abs(Dmax) / DincrStatic)

        # Add the incremental displacements for the first peak.
        for i in range(int(abs(Dmax) / DincrStatic)):
            iDstep[start_index + i] = (i + 1) * DincrStatic

        # Add the incremental displacements for the second peak, if necessary.
        if CycleType != "Push":
            for i in range(int(abs(Dmax) / DincrStatic)):
                iDstep[start_index + int(abs(Dmax) / DincrStatic) + i] = iDstep[start_index + int(abs(Dmax) / DincrStatic) - 1 - i]

            # Add the incremental displacements for the third peak, if necessary.
            if CycleType != "Half":
                for i in range(int(abs(Dmax) / DincrStatic)):
                    iDstep[start_index + 2 * int(abs(Dmax) / DincrStatic) + i] = iDstep[start_index + 2 * int(abs(Dmax) / DincrStatic) - 1 - i] * -1

                # Add the incremental displacements for the fourth peak, if necessary.
                for i in range(int(abs(Dmax) / DincrStatic)):
                    iDstep[start_index + 3 * int(abs(Dmax) / DincrStatic) + i] = iDstep[start_index + 3 * int(abs(Dmax) / DincrStatic) - 1 - i]

    return iDstep


# Vector of displacement-cycle peaks in terms of wall drift ratio (flexural
# displacements).
iDmax = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Dincr = 0.2
CycleType = "Full"
Ncycles = 3

# Generate the incremental displacements for each peak.
iDstep = []
for Dmax in iDmax:
    iDstep.append(generate_peaks(Dmax, Dincr, CycleType, Ncycles=Ncycles))

# Flatten the list of lists into a single list
iDstep_flat = [item for sublist in iDstep for item in sublist]

# Specify the filename where you want to save the data
output_file = "incremental_displacements.csv"

# Save the data to a CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Incremental Displacements"])  # Add a header
    writer.writerows([[item] for item in iDstep_flat])  # Save each value in a single column

print(f"Data has been saved to {output_file}")
