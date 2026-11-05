Numerical PDEs & Immersed Boundary Methods

Scientific Computing | Numerical Analysis | Computational Mathematics

A collection of Python implementations exploring numerical solutions of partial differential equations, immersed boundary methods, sparse linear algebra, and iterative solver performance.

This project investigates the numerical solution of elliptic PDEs on Cartesian grids, extending classical finite-difference methods to problems involving complex geometries, internal boundary constraints, and dynamically evolving immersed boundaries.

The implementations emphasize numerical accuracy, computational efficiency, convergence verification, and scientific visualization.

Technologies: Python · NumPy · SciPy · Matplotlib · Sparse Linear Algebra · GMRES

⸻

Project Overview

The project consists of four numerical experiments, progressing from classical finite-difference discretizations to moving immersed boundary simulations.

Project	Numerical Focus	Key Techniques
01 — Finite-Difference Poisson Solver	Numerical solution of elliptic PDEs	Five-point stencil, sparse matrices, convergence analysis
02 — Immersed Boundary Poisson Solver	Internal geometric constraints	Regularized delta functions, spreading/interpolation operators
03 — Constrained Surface Reconstruction	Elliptic PDEs with prescribed boundary values	Homogeneous Poisson equation, immersed boundary forcing
04 — Dynamic Immersed Boundary Simulation	Moving geometries and solver optimization	GMRES, warm-start strategies, 2D/3D animation

Each implementation is contained in a standalone Python script, with corresponding numerical results and visualizations.

⸻

01. Finite-Difference Poisson Solver

Implementation: P1.py

Mathematical Model

Consider the two-dimensional Poisson equation

$$
\nabla^2 u(x,y)=f(x,y),
$$

defined on the computational domain

$$
\Omega=[0,2\pi]^2.
$$

The Laplacian is discretized using the standard five-point finite-difference approximation:

$$
\nabla_h^2u_{i,j}

\frac{
u_{i+1,j}+u_{i-1,j}
+u_{i,j+1}+u_{i,j-1}
-4u_{i,j}
}{h^2}.
$$

This produces a sparse linear system

$$
A\mathbf{u}=\mathbf{f},
$$

where $A$ represents the discrete Laplacian.

The discretization has a local truncation error of

$$
\mathcal{O}(h^2).
$$

Numerical Verification

To verify the implementation, the numerical solution is compared against a known analytical solution.

Relative errors are measured using the discrete infinity and Euclidean norms:

$$
E_\infty=
\frac{|u_h-u_{\mathrm{exact}}|\infty}
{|u{\mathrm{exact}}|_\infty},
$$

$$
E_2=
\frac{|u_h-u_{\mathrm{exact}}|2}
{|u{\mathrm{exact}}|_2}.
$$

Convergence Results

Both error norms closely follow the theoretical second-order convergence rate:

$$
\boxed{E(h)=\mathcal{O}(h^2)}
$$

The results provide numerical verification of the finite-difference discretization and its expected accuracy under grid refinement.

Key concepts: Finite differences, sparse matrix assembly, truncation error, numerical convergence, error norms.

⸻

02. Immersed Boundary Poisson Solver

Implementation: P2.py

Motivation

Traditional finite-difference methods are particularly convenient on structured Cartesian grids. However, representing curved or irregular internal boundaries introduces additional challenges.

The immersed boundary method addresses this by representing geometric constraints independently of the underlying Cartesian discretization.

This implementation extends the Poisson solver to incorporate a circular immersed boundary using regularized delta functions.

Mathematical Formulation

The governing equation is

$$
\nabla^2u=f+S^{ib}(q),
$$

subject to the immersed boundary constraint

$$
J^{ib}(u)=g.
$$

The spreading operator $S^{ib}$ transfers forces defined along the immersed boundary onto the Cartesian grid.

The interpolation operator $J^{ib}$ evaluates the Cartesian solution at immersed boundary locations.

The immersed boundary is parameterized as

$$
X(\theta)=
\begin{bmatrix}
\pi+\cos\theta\
\pi+\sin\theta
\end{bmatrix},
\qquad
\theta\in[0,2\pi].
$$

Following discretization, the coupled system takes the block form

$$
\begin{bmatrix}
A & -S\
J & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{u}\
\mathbf{q}
\end{bmatrix}

\begin{bmatrix}
\mathbf{f}\
\mathbf{g}
\end{bmatrix}.
$$

This formulation simultaneously solves for the Cartesian-grid solution and the unknown immersed boundary forces.

Convergence Analysis

The numerical solution is evaluated across multiple Cartesian grid resolutions.

The infinity-norm error closely follows the second-order reference curve, while the relative Euclidean error decreases consistently under refinement.

These experiments illustrate the accuracy of the immersed boundary formulation for the tested geometry.

Boundary Resolution and Computational Cost

An additional numerical experiment investigates the relationship between immersed boundary point spacing and Cartesian grid resolution.

The objective is to determine how boundary discretization influences accuracy and computational expense.

The results show that excessively dense immersed boundary discretizations can increase computational cost without necessarily improving solution accuracy.

For the tested configuration, immersed boundary spacing comparable to the Cartesian grid spacing provides a practical balance between accuracy and efficiency.

Key concepts: Immersed boundary methods, regularized delta functions, interpolation, force spreading, coupled linear systems, resolution studies.

⸻

03. Constrained Surface Reconstruction

Implementation: P3.py

Mathematical Formulation

This experiment investigates the construction of a smooth surface satisfying prescribed values along an internal boundary.

The governing equation is the homogeneous Poisson equation with immersed boundary forcing:

$$
\nabla^2u=S^{ib}(q).
$$

Homogeneous Dirichlet conditions are imposed on the outer computational boundary:

$$
u|_{\partial\Omega}=0.
$$

Along the circular immersed boundary, the prescribed values are

$$
u(X(\theta))

\cos(2X_1(\theta))
\sin(2X_2(\theta)).
$$

The solution is harmonic away from the immersed boundary forcing and is constrained by the prescribed internal and external boundary values.

Numerical Interpretation

The resulting numerical surface develops localized positive and negative deformations corresponding to the oscillatory immersed boundary data.

Away from the immersed boundary, the solution smoothly transitions toward the homogeneous outer boundary conditions.

This provides a computational example of how elliptic PDEs can be used to construct smooth surfaces under geometric constraints.

The resulting surface is described as almost minimal: although it exhibits smooth interpolation between prescribed values, it is governed by the linear Laplace equation rather than the nonlinear minimal-surface equation.

Key concepts: Elliptic PDEs, harmonic functions, constrained surfaces, Dirichlet conditions, immersed boundary forcing.

⸻

04. Dynamic Immersed Boundary Simulation

Implementation: P4.py

This experiment extends the immersed boundary framework to multiple moving and deforming geometries.

It also investigates the performance of iterative linear solvers when solving a sequence of related numerical systems.

Moving Boundary Geometry

Two independently evolving immersed boundaries are introduced:

* Rotating ellipse: An elliptical boundary undergoing continuous rotation.
* Deforming circle: A circular boundary undergoing periodic geometric deformation.

At each simulation frame, the immersed boundary geometry is updated and the corresponding elliptic PDE is solved.

The resulting system can be represented as

$$
\mathcal{L}_h u^{(k)}
+
\mathcal{S}^{(k)}F^{(k)}
=f,
$$

where:

* $\mathcal{L}_h$ is the discrete Laplacian.
* $\mathcal{S}^{(k)}$ is the geometry-dependent spreading operator.
* $F^{(k)}$ represents immersed boundary forces.
* $k$ denotes the simulation frame.

Because the elliptic problem is recomputed independently at each configuration, the simulation is pseudo-dynamic rather than a direct time integration of a time-dependent PDE.

2D Numerical Simulation

The animation visualizes the evolution of the numerical solution as the immersed boundaries rotate and deform.

The black curves represent the immersed boundaries, while the contour field represents the computed solution.

Changes in boundary geometry produce corresponding changes in the surrounding numerical field.

3D Surface Evolution

The three-dimensional representation illustrates the evolving numerical surface.

Localized peaks, valleys, and complex spatial structures emerge as the immersed boundaries move through successive configurations.

These animations demonstrate how geometric constraints can influence solutions of elliptic PDEs.

⸻

Iterative Solver Performance: GMRES

A separate computational experiment investigates whether information from previous numerical solutions can accelerate subsequent linear solves.

The generalized minimal residual method (GMRES) is used to solve the resulting linear systems.

Two initialization strategies are compared.

Cold Start

Each solve begins with a zero initial guess:

$$
u_0^{(k)}=0.
$$

Warm Start

The previous numerical solution is reused as the initial guess:

$$
u_0^{(k)}=u^{(k-1)}.
$$

Because consecutive immersed boundary configurations are geometrically similar, their corresponding numerical solutions may also be close.

The warm-start strategy attempts to exploit this relationship to reduce the computational work required for convergence.

Performance Results

The experiments compare GMRES iteration counts and solution times across successive simulation frames.

Observed results:

* Both initialization strategies require approximately 50–54 GMRES iterations per frame.
* Warm-starting does not consistently reduce the number of Krylov iterations.
* Solution times remain comparable, with warm-starting occasionally performing slightly worse.
* The previous numerical solution does not provide a substantial computational advantage for this particular system.

Interpretation

Although warm-starting is a natural strategy for sequences of related linear systems, its effectiveness depends on more than the proximity of consecutive solutions.

GMRES convergence is influenced by the initial residual, the linear operator, and the spectral properties of the system.

The results demonstrate that a seemingly improved initial approximation does not necessarily translate into fewer iterations or lower computational cost.

For this experiment, alternative preconditioning strategies may provide a more effective direction for improving solver performance.

This comparison emphasizes the importance of benchmarking numerical algorithms rather than assuming that theoretically motivated modifications will improve practical performance.

⸻

Technical Implementation

Numerical Methods

The project implements and investigates:

Category	Methods
PDE Discretization	Finite differences, five-point Laplacian
Numerical Linear Algebra	Sparse matrices, coupled block systems
Iterative Solvers	GMRES, warm-start and cold-start initialization
Immersed Boundary Methods	Force spreading, interpolation, regularized delta functions
Numerical Verification	Grid refinement, relative error norms, convergence rates
Computational Experiments	Boundary spacing studies, runtime comparisons
Scientific Visualization	Contour plots, 3D surfaces, animated simulations

Software Stack

Language: Python

Libraries:

* NumPy: Array operations and numerical computation.
* SciPy: Sparse linear algebra and numerical solvers.
* Matplotlib: Scientific visualization and animation.

⸻

Repository Structure

Numerical-PDEs-Immersed-Boundary-Methods/
│
├── P1.py                      # Finite-difference Poisson solver
├── P2.py                      # Immersed boundary Poisson solver
├── P3.py                      # Constrained surface reconstruction
├── P4.py                      # Dynamic immersed boundary simulation
│
├── Figures/                   # Numerical results and convergence plots
│
├── Animations/                # 2D and 3D simulations
│   ├── P4_2D_Animation.gif
│   └── P4_3D_Animation.gif
│
├── Report/                    # Mathematical derivations and analysis
│
├── .gitignore
└── README.md

⸻

Installation and Execution

Requirements

* Python 3
* NumPy
* SciPy
* Matplotlib

Install the required Python packages:

python -m pip install numpy scipy matplotlib

Running the Implementations

Each numerical experiment can be executed independently.

Finite-Difference Poisson Solver

python P1.py

Immersed Boundary Poisson Solver

python P2.py

Constrained Surface Reconstruction

python P3.py

Dynamic Immersed Boundary Simulation

python P4.py

The scripts generate numerical solutions, convergence studies, and scientific visualizations.

The dynamic immersed boundary experiment includes repeated linear solves and animation generation, so execution may take longer.

⸻

Key Results

This project demonstrates several fundamental ideas in numerical PDEs and scientific computing.

1. Numerical convergence verification

The five-point finite-difference discretization achieves the expected second-order convergence rate, validating the implementation against an analytical solution.

2. Geometric flexibility

Immersed boundary methods allow internal geometric constraints to be incorporated into structured Cartesian-grid solvers.

3. Accuracy versus computational cost

Immersed boundary discretization density influences both solution accuracy and computational expense, motivating careful resolution selection.

4. Dynamic geometry

Moving and deforming immersed boundaries can be handled through sequences of elliptic PDE solves, producing evolving numerical fields and surfaces.

5. Iterative solver benchmarking

Warm-starting GMRES does not necessarily reduce computational cost. Numerical performance must be evaluated empirically, and preconditioning may be a more promising optimization strategy.

⸻

Mathematical Documentation

Additional mathematical formulations, implementation details, convergence studies, and numerical discussions are available in the Report/ directory.

⸻

Author

Axel Fraud

Ph.D. Student, Computational Applied Mathematics
Colorado School of Mines

Research interests: Numerical Optimization · Scientific Computing · Numerical Linear Algebra · Computational Mathematics · Machine Learning