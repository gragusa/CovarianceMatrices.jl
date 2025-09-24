import numpy as np
from pathlib import Path

# Configuration parameters
SEED = 20240520
T = 1000  # number of time periods
P = 5     # dimension of the series
MAX_EIGENVALUE = 0.8

rng = np.random.default_rng(SEED)

# Draw a random matrix and scale it so that its spectral radius is MAX_EIGENVALUE
b_raw = rng.normal(size=(P, P))
maxeig_raw = np.abs(np.linalg.eigvals(b_raw)).max()
scale = MAX_EIGENVALUE / max(maxeig_raw, 1e-12)
B = b_raw * scale

# Simulate MA(1): DAT_t = U_t + B U_{t-1}
U = np.zeros((T + 1, P))
U[1:] = rng.normal(size=(T, P))

DAT = np.zeros((T, P))
for t in range(1, T + 1):
    DAT[t - 1] = U[t] + B @ U[t - 1]

# Save the data file (NT, KDIM on first line, then rows of DAT)
data_path = Path("dat_ma1.txt")
with data_path.open("w", encoding="ascii") as f:
    f.write(f"{T} {P}\n")
    for row in DAT:
        f.write(" ".join(f"{value:.10f}" for value in row) + "\n")

# Save metadata about the simulation
info_path = Path("ma1_info.txt")
with info_path.open("w", encoding="ascii") as f:
    f.write(f"Seed: {SEED}\n")
    f.write(f"Scale applied to raw B: {scale:.10f}\n")
    f.write("B matrix (post-scale):\n")
    for row in B:
        f.write(" ".join(f"{value:.10f}" for value in row) + "\n")
    f.write("Eigenvalues of B (post-scale):\n")
    for eigenvalue in np.linalg.eigvals(B):
        f.write(f"{eigenvalue.real: .10f} {eigenvalue.imag: .10f}\n")

print("Generated", data_path)
print("Generated", info_path)
