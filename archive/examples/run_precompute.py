import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data points (approximate, matching the original figure structure)
years = [int(2010), int(2012), int(2017), int(2019), int(2019), int(2025), int(2026)]
labels = [
    "Jones et al.",
    "Fowler et al.",
    "O'Gorman et al.",
    "Gheorghiu et al.",
    "Gidney et al.",
    "Gidney et al. 2025",
    "Atlantic Quantum",
]
physical_qubits = [
    5e8,   # Jones et al.
    1e9,   # Fowler et al.
    2e8,   # O'Gorman et al.
    1.5e8, # Gheorghiu et al.
    2e7,   # Gidney et al. (earlier)
    1e6,   # Gidney et al. 2025
    1e5,   # Atlantic Quantum
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(years, physical_qubits)

# Log scale for y-axis
plt.yscale("log")

# Labels
plt.xlabel("Year")
plt.ylabel("Physical Qubits")

# Annotate each point
for x, y, label in zip(years, physical_qubits, labels):
    plt.text(x +0.15, y, f" {label}", fontsize=9, verticalalignment="center")

# Grid
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim(2009, 2030)
plt.tight_layout()
plt.show()