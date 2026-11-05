Numerical PDEs & Immersed Boundary Methods

Python | Scientific Computing | Numerical Analysis | Sparse Linear Algebra

A collection of Python implementations for solving partial differential equations using finite-difference and immersed boundary methods. This project explores numerical accuracy, computational efficiency, iterative linear solvers, and scientific visualization for elliptic PDEs with complex and evolving geometries.

Visualizations

Dynamic Immersed Boundaries — 3D Surface Evolution

Moving and Deforming Boundaries — 2D Simulation

Numerical Implementations

Module	Description	Methods
P1.py	2D Poisson equation solver	Five-point finite differences, sparse matrices, convergence analysis
P2.py	Poisson solver with embedded geometric constraints	Immersed boundary spreading/interpolation, regularized delta functions
P3.py	Constrained surface reconstruction	Homogeneous Poisson equation, prescribed immersed boundary values
P4.py	Moving-boundary simulations	Dynamic geometry, GMRES, iterative solver benchmarking

Numerical Results

Finite-Difference Convergence

The two-dimensional Poisson solver uses the standard five-point Laplacian,

$$
\nabla_h^2 u_{i,j}

\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2},
$$

and demonstrates the expected second-order convergence under grid refinement.

Immersed Boundary Methods

The immersed boundary formulation incorporates internal geometric constraints into a Cartesian-grid PDE solver through discrete spreading and interpolation operators:

$$
\nabla^2u=f+S^{ib}(q),
\qquad
J^{ib}(u)=g.
$$

Numerical experiments examine convergence behavior and the relationship between immersed boundary resolution, accuracy, and computational cost.

GMRES Performance Analysis

For moving-boundary simulations, GMRES is evaluated using two initialization strategies:

* Cold start: Initialize each linear solve with zero.
* Warm start: Reuse the previous solution as the initial guess.

The experiments show comparable convergence behavior, with approximately 50–54 iterations per frame. Warm-starting provides no significant computational advantage for the tested system.

Technologies & Methods

* Python: NumPy, SciPy, Matplotlib
* Numerical PDEs: Finite differences, elliptic equations, immersed boundary methods
* Linear Algebra: Sparse matrix systems, GMRES
* Numerical Verification: Grid refinement, convergence rates, relative error norms
* Scientific Visualization: 2D contour plots, 3D surfaces, animated simulations

Installation & Usage

Install dependencies:

python -m pip install numpy scipy matplotlib

Run each implementation independently:

python P1.py
python P2.py
python P3.py
python P4.py

Repository Structure

├── P1.py               # Finite-difference Poisson solver
├── P2.py               # Immersed boundary solver
├── P3.py               # Constrained surface reconstruction
├── P4.py               # Dynamic immersed boundary simulation
├── Figures/            # Convergence plots and numerical results
├── Animations/         # 2D and 3D simulations
├── Report/             # Mathematical derivations and analysis
└── README.md

Documentation

Detailed mathematical formulations, convergence studies, numerical experiments, and performance analysis are available in the technical report.

⸻

Axel Fraud
Ph.D. Student, Computational Applied Mathematics
Colorado School of Mines
