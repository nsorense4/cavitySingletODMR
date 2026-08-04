# cavitySingletODMR

Analysis and simulation code for cavity-coupled NV singlet ODMR / IR magnetometry:
Lindblad master-equation modeling of the NV 11-level system, cavity transmission
fitting, and IR-based magnetometry sensitivity analysis.

## Layout

- `__python/` — active analysis scripts
- `__matlab/` — MATLAB scripts for transmission data fitting/plotting
- `__figures/` — generated figure outputs
- `__refFiles/` — precomputed lookup tables / fit results (`.npz`)
- `__deprecated/` — superseded scripts, kept for reference only, not maintained

## Setup

```bash
pip install -r requirements.txt
```

Pins [commonlabcode](https://github.com/nsorense4/commonLabCode) at `v0.1.0` for shared
plotting utilities. Bump the tag in `requirements.txt` deliberately if you want a newer
version.

Several scripts (`linbladMESolverFunctionChargeConversionLiouvillian.py`,
`linbladMESolverFunctionChargeConversion.py`) depend on `qutip`, and use `numba` for
JIT-accelerated rate solving — the numba cache (`__pycache__/*.nbc`, `*.nbi`) is
gitignored since it's machine-specific.

Many scripts load raw `.mat`/experimental data from absolute paths on the BarclayLab
OneDrive; they will only run on a machine with access to that data.
