# MiniOps demos

These notebooks are organized as a short learning sequence. They use **CairoMakie** consistently for figures; plotting is not part of the MiniOps operator API itself.

1. `00_getting_started.ipynb` — basic operators, adjoints, diagnostics, composition, FFT.
2. `01_sampling_and_composition.ipynb` — scalar/trace sampling and composite operators.
3. `02_deconvolution_and_solvers.ipynb` — CGLS and ISTA on a convolution inverse problem.
4. `03_sparse_fourier_reconstruction.ipynb` — 2-D and 3-D sparse Fourier reconstruction.
5. `04_radon_inversion.ipynb` — parabolic Radon modeling and sparse inversion.
6. `05_blending.ipynb` — blending and adjoint gathering.

## Running a notebook

Each notebook activates the MiniOps project with `Pkg.activate("..")`. After cloning or moving the repository, run `Pkg.instantiate()` once if the project dependencies are not already installed. The notebooks intentionally do not call `Pkg.add` or modify the environment.
