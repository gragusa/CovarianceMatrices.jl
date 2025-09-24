# VARHAC Simulation Notes

## Generated files
- `generate_ma1_data.py` – Python script that simulates the MA(1) dataset and writes the supporting metadata. Run it to regenerate `dat_ma1.txt` and `ma1_info.txt`.
- `dat_ma1.txt` – simulated data matrix used as input to the VARHAC routine.
- `ma1_info.txt` – diagnostic information about the MA(1) simulation (scale factor, MA matrix `B`, eigenvalues).
- `varhac_run_imodel1.txt` – VARHAC output using AIC model selection (`IMODEL=1`), maximum lag 4 (`IMAX=4`), and mean subtraction enabled (`IMEAN=1`).
- `varhac_run_imodel2.txt` – VARHAC output using BIC model selection (`IMODEL=2`), `IMAX=4`, `IMEAN=1`.
- `varhac_run_imodel3.txt` – VARHAC output with fixed lag order equal to `IMAX=4` (`IMODEL=3`), `IMEAN=1`.
- `varhac_run_imodel1_imean0.txt` – VARHAC output using AIC selection (`IMODEL=1`), `IMAX=4`, but without mean removal (`IMEAN=0`).

## File formats
### `dat_ma1.txt`
1. First line contains two integers: the number of observations `NT` and the number of series `KDIM` (here: `1000 5`).
2. The next `NT` lines each hold `KDIM` floating-point numbers representing one row of the simulated data matrix `DAT`.

### `ma1_info.txt`
Plain-text human-readable summary:
- Random seed used for the simulation.
- Scaling factor applied to the initial random MA matrix.
- The scaled MA matrix `B`, one row per line.
- Eigenvalues of `B`, each line giving the real and imaginary parts.

### `varhac_run_*.txt`
Each run file shares the same structure:
1. Five lines showing the upper-left 5x5 block of the intermediate `ATEMP` matrix printed by the legacy FORTRAN routine (numeric values in fixed-width format).
2. A run summary block with labelled fields for `Data file`, `NT`, `KDIM`, `IMODEL`, `IMAX`, and `IMEAN`.
3. The estimated variance-covariance matrix `AAA`, printed as `KDIM` lines with scientific-notation entries.

Interpretation of VARHAC parameters:
- `IMODEL=1`: Akaike Information Criterion (AIC) selects the lag order for each series.
- `IMODEL=2`: Bayesian Information Criterion (BIC) selects the lag order.
- `IMODEL=3`: Uses a fixed lag order equal to `IMAX` (no information criterion).
- `IMAX`: Maximum lag order explored (or the fixed order when `IMODEL=3`).
- `IMEAN=1`: The mean of each series is removed before estimation. `IMEAN=0` leaves the data unchanged.

## Python helper snippets
### Load the simulated data matrix
```python
import numpy as np

def load_dat_matrix(path="dat_ma1.txt"):
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().strip().split()
        nt, kdim = map(int, header)
        data = np.loadtxt(f)
        data = data.reshape(nt, kdim)
    return nt, kdim, data
```

### Parse a VARHAC output file
```python
def load_varhac_output(path):
    entries = []
    meta = {}
    with open(path, "r", encoding="ascii") as f:
        for _ in range(5):
            entries.append([float(x) for x in f.readline().split()])
        f.readline()  # --- VARHAC run ---
        meta["data_file"] = f.readline().split(":", 1)[1].strip()
        meta["NT"] = int(f.readline().split(":", 1)[1])
        meta["KDIM"] = int(f.readline().split(":", 1)[1])
        meta["IMODEL"] = int(f.readline().split(":", 1)[1])
        meta["IMAX"] = int(f.readline().split(":", 1)[1])
        meta["IMEAN"] = int(f.readline().split(":", 1)[1])
        f.readline()  # AAA matrix:
        matrix = []
        for _ in range(meta["KDIM"]):
            matrix.append([float(x) for x in f.readline().split()])
    return np.array(entries), meta, np.array(matrix)
```

### Example usage
```python
_, _, dat = load_dat_matrix()
_, params, cov = load_varhac_output("varhac_run_imodel1.txt")
print("IMODEL:", params["IMODEL"])
print("AAA[0,0]:", cov[0, 0])
```
