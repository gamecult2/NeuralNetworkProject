
# Data from the table
data = {
    "EXP-L-S1-A0": {"Pmax": 93.69, "Pmean": 32.72, "δ": 88.00},
    "EXP-L-S1-A1": {"Pmax": 110.69, "Pmean": 18.14, "δ": 66.00},
    "EXP-L-S1-A2": {"Pmax": 187.12, "Pmean": 99.72, "δ": 32.00},
    "EXP-L-S1-A3": {"Pmax": 217.13, "Pmean": 131.76, "δ": 27.00},
    "EXP-L-S2-A0": {"Pmax": 133.07, "Pmean": 49.14, "δ": 59.00},
    "EXP-L-S2-A1": {"Pmax": 194.16, "Pmean": 45.90, "δ": 27.00},
    "EXP-L-S2-A2": {"Pmax": 242.35, "Pmean": 82.12, "δ": 19.00},
    "EXP-L-S1.5-A0": {"Pmax": 90.76, "Pmean": 38.30, "δ": 68.00},
    "EXP-L-S1.5-A3": {"Pmax": 235.04, "Pmean": 158.94, "δ": 24.00},
}

# Calculate percentage reduction for thicknesses 1, 2, and 3 mm
thicknesses = [1, 2, 3]
for thickness in thicknesses:
    reference_length = data["EXP-L-S1-A0"]["δ"]  # Reference length (0 mm thickness)
    reduction_mm = reference_length - data[f"EXP-L-S1-A{thickness}"]["δ"]
    reduction_percent = (reduction_mm / reference_length) * 100
    print(f"Thickness {thickness} mm: {reduction_percent:.2f}%")