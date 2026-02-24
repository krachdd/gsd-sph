# General Simulation Data Format for SPH Solver

**Version:** 3.4.2 (based on [GSD by Glotzer Group](https://github.com/glotzerlab/gsd))

A subproject of [hoomd-sph3](https://www.mib.uni-stuttgart.de/institute/team/Krach/)
that extends the upstream GSD library with the particle fields required by the SPH
solver. All modifications are confined to the Python API (`gsd/hoomd.py`); the C file
layer (`gsd/gsd.c`, `gsd/gsd.h`) is unmodified.

---

## SPH-specific particle fields

The following fields are added to `gsd.hoomd.ParticleData` on top of the standard
HOOMD schema:

| Field | Shape | dtype | Description |
|---|---|---|---|
| `slength` | (*N*,) | `float32` | Smoothing length *h* |
| `density` | (*N*,) | `float32` | SPH density *ρ* |
| `pressure` | (*N*,) | `float32` | Pressure *p* |
| `energy` | (*N*,) | `float32` | Specific internal energy *e* |
| `auxiliary1` | (*N*, 3) | `float32` | User-defined vector field 1 |
| `auxiliary2` | (*N*, 3) | `float32` | User-defined vector field 2 |
| `auxiliary3` | (*N*, 3) | `float32` | User-defined vector field 3 |
| `auxiliary4` | (*N*, 3) | `float32` | User-defined vector field 4 |

Fields unused by SPH (`orientation`, `angmom`, `charge`, `diameter`,
`moment_inertia`) have been removed from the schema defaults to keep output files
compact.

---

## Usage

```python
import gsd.hoomd
import numpy as np

N = 100

# --- write ---
frame = gsd.hoomd.Frame()
frame.configuration.box = [10, 10, 10, 0, 0, 0]
frame.particles.N = N
frame.particles.position  = np.zeros((N, 3), dtype=np.float32)
frame.particles.velocity  = np.zeros((N, 3), dtype=np.float32)
frame.particles.mass      = np.ones(N, dtype=np.float32)
frame.particles.slength   = np.full(N, 0.1, dtype=np.float32)
frame.particles.density   = np.full(N, 1000.0, dtype=np.float32)
frame.particles.pressure  = np.zeros(N, dtype=np.float32)
frame.particles.energy    = np.zeros(N, dtype=np.float32)

with gsd.hoomd.open('trajectory.gsd', 'w') as traj:
    traj.append(frame)

# --- read ---
with gsd.hoomd.open('trajectory.gsd', 'r') as traj:
    f = traj[0]
    print(f.particles.density)
    print(f.particles.slength)
```

---

## Build

The conda environment required for this project is defined in the parent
`hoomd-sph3` repository.

```bash
cd gsd-3.4.2
mkdir build && cd build
cmake ..
make
```

After building, install the Python package into the active environment:

```bash
pip install -e gsd-3.4.2/
```

---

## Developer

[David Krach](https://www.mib.uni-stuttgart.de/institute/team/Krach/)
— [david.krach@mib.uni-stuttgart.de](mailto:david.krach@mib.uni-stuttgart.de)
University of Stuttgart, Institute of Applied Mechanics (MIB)
