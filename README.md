# CFD Mini-Portfolio — C++ & OpenFOAM
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-2312-8A2BE2)]()
[![Numerics](https://img.shields.io/badge/Numerics-FDM/FVM-green)]()

<p align="center">
  <img src="RB1.gif" alt="Rayleigh–Bénard convection" width="320">
  <img src="RB1_FTLE.gif" alt="Finite-time Lyapunov exponent RB" width="320">
  <img src="cavity-re100.gif" alt="Lid-driven cavity Re=100" width="320">
</p>

---

## Contents
- [Lid-Driven Cavity (C++)](#lid-driven-cavity)
- [Channel Flow (C++)](#channel-flow)
- [Backwards Facing Step (C++)](#channel-flow)
- [Rayleigh–Bénard Convection](#rayleigh–bénard-convection)
- [Inlet Box (OpenFOAM)](#inlet-box-openfoam)
- [Run It / Reproducibility](#run-it--reproducibility)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Lid-Driven Cavity (C++)
2D unsteady incompressible cavity; projection method; centerline validation vs. Ghia et al.

<p align="center">
  <img src="cavity-re100.png" alt="Cavity velocity contours, Re=100" width="480">
</p>

<details>
<summary><b>Details</b></summary>

- **Discretization**: FDM on a uniform grid, 2nd-order central; projection step via Poisson solve.
- **BCs**: No-slip walls; moving lid.
- **Outputs**: centerline profiles, residual logs, PNG frames.
- **Todo**: switch to multigrid Poisson, add Re=1000 stability notes.

</details>

---

## Channel Flow (C++)
2D channel; start-up transient to fully developed profile; compares with analytic Poiseuille.

<p align="center">
  <img src="channel-re100.png" alt="Channel flow velocity profile" width="480">
</p>

<details>
<summary><b>Details</b></summary>

- **Inlet**: fixed U or flow-rate; **Outlet**: ∂/∂x = 0.
- **Validation**: parabolic profile; friction factor vs. Re.

</details>

---

## Backwards Facing Step (C++)
2D channel with a step at the inlet

<p align="center">
  <img src="backwards_step-re100.png" alt="Backwards Facing Step" width="480">
</p>

<details>
<summary><b>Details</b></summary>

- **Inlet**: fixed U or flow-rate; **Outlet**: ∂/∂x = 0.
- **Validation**: parabolic profile; friction factor vs. Re.

</details>

---

## Rayleigh–Bénard Convection
Thermal convection visualizations and diagnostics (FTLE).

<p align="center">
  <img src="RB1.gif" alt="Rayleigh–Bénard convection animation" width="360">
  <img src="RB1_FTLE.gif" alt="FTLE ridges for RB" width="360">
</p>

<details>
<summary><b>Details</b></summary>

- **Quantities**: temperature, vorticity, streamfunction, **FTLE**.
- **Knobs**: Ra, Pr, grid size.
- **Todo**: Nu vs. Ra plot; spectral vs. FD comparison.

</details>

---

## Industrial Ductwork (OpenFOAM)
Industrial duct elbow; tetra mesh (~300k); streamlines + pressure contours.

<table>
<tr>
<td><img src="mesh.png" alt="Inlet box streamlines" width="420"></td>
<td><img src="Velocity-streamlines-contour.png" alt="Inlet box streamlines" width="420"></td>
<td><img src="pressure-contours.png" alt="Inlet box streamlines" width="420"></td>
</tr>
</table>

<details>
<summary><b>Case Notes</b></summary>

- **Meshing**: SALOME → UNV → `ideasUnvToFoam`; wall y⁺ target; SST k-ω.
- **Solver**: `pimpleFoam`; `adjustTimeStep yes; maxCo 0.7`.
- **Parallel**: `scotch`, `mpirun -np 8`.
- **Todo**: vane param study; Δp vs. flow rate curve.

</details>

---

## Run It / Reproducibility
```bash
# C++ examples (cavity/channel)
mkdir -p build && cd build && cmake .. && make -j
./cavity   --Re 100   --Nx 128 --Ny 128 --dt 1e-3
./channel  --Re 1000  --Nx 256 --Ny 64  --dt 5e-4

# OpenFOAM inlet box (example)
decomposePar -force
mpirun -np 8 pimpleFoam -parallel
reconstructPar -latestTime
