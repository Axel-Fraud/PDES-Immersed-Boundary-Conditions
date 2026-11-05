import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


# ============================================================
# MA550 - Homework 2
# Problem 1: Poisson Equation
# ============================================================

# Domain: [0, 2*pi] x [0, 2*pi]
L = 2 * np.pi


# ------------------------------------------------------------
# Manufactured exact solution and forcing function
# ------------------------------------------------------------

def u_exact(x, y):
    return np.cos(2 * x) * np.sin(2 * y)


def forcing(x, y):
    return -8 * np.cos(2 * x) * np.sin(2 * y)


# ------------------------------------------------------------
# Finite difference solver
# ------------------------------------------------------------

def poisson_solver(N):

    # Uniform Cartesian grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Exact solution and forcing
    U_exact = u_exact(X, Y)
    F = forcing(X, Y)

    # Number of interior grid points
    n = N - 2

    # 1D second derivative matrix
    T = diags(
        [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr"
    ) / h**2

    I = eye(n, format="csr")

    # 2D Laplacian using Kronecker products
    A = kron(I, T) + kron(T, I)

    # Right-hand side
    b = F[1:-1, 1:-1].copy()

    # Dirichlet boundary conditions
    b[0, :] -= U_exact[0, 1:-1] / h**2
    b[-1, :] -= U_exact[-1, 1:-1] / h**2

    b[:, 0] -= U_exact[1:-1, 0] / h**2
    b[:, -1] -= U_exact[1:-1, -1] / h**2

    # Solve sparse linear system
    u_interior = spsolve(A, b.ravel()).reshape((n, n))

    # Construct full numerical solution
    U = U_exact.copy()
    U[1:-1, 1:-1] = u_interior

    return X, Y, U, U_exact, h


# ============================================================
# Part 1: Numerical Solution
# ============================================================

N = 81

X, Y, U, U_exact, h = poisson_solver(N)

error = np.abs(U - U_exact)

print("=" * 60)
print("POISSON EQUATION - NUMERICAL SOLUTION")
print("=" * 60)

print(f"Grid points per direction: N = {N}")
print(f"Grid spacing: Delta x = {h:.6f}")
print(f"Maximum absolute error: {np.max(error):.6e}")


# ------------------------------------------------------------
# 2D visualization
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

contour = ax.contourf(
    X, Y, U,
    levels=50,
    cmap="cividis"
)

fig.colorbar(contour, ax=ax, label=r"$u(x,y)$")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Finite-Difference Solution of the Poisson Equation")

plt.tight_layout()
plt.savefig("P1_2D.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# 3D visualization
# ------------------------------------------------------------

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, U,
    cmap="cividis",
    edgecolor="none",
    antialiased=True
)

fig.colorbar(
    surf,
    ax=ax,
    shrink=0.65,
    pad=0.1,
    label=r"$u(x,y)$"
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u(x,y)$")

ax.set_title("Finite-Difference Solution of the Poisson Equation")

plt.tight_layout()
plt.savefig("P1_3D.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Part 2: Convergence Study
# ============================================================

N_values = np.array([11, 21, 41, 81, 161])

h_values = []
errors_inf = []
errors_2 = []

for N in N_values:

    X, Y, U, U_exact, h = poisson_solver(N)

    E = U - U_exact

    # Relative infinity norm
    error_inf = np.max(np.abs(E)) / np.max(np.abs(U_exact))

    # Relative L2 norm
    error_2 = (
        np.linalg.norm(E.ravel(), 2)
        / np.linalg.norm(U_exact.ravel(), 2)
    )

    h_values.append(h)
    errors_inf.append(error_inf)
    errors_2.append(error_2)

h_values = np.array(h_values)
errors_inf = np.array(errors_inf)
errors_2 = np.array(errors_2)


# ------------------------------------------------------------
# Observed convergence orders
# ------------------------------------------------------------

orders_inf = (
    np.log(errors_inf[:-1] / errors_inf[1:])
    / np.log(h_values[:-1] / h_values[1:])
)

orders_2 = (
    np.log(errors_2[:-1] / errors_2[1:])
    / np.log(h_values[:-1] / h_values[1:])
)


# ------------------------------------------------------------
# Print convergence results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONVERGENCE STUDY")
print("=" * 60)

print(
    f"{'N':>6} "
    f"{'Delta x':>12} "
    f"{'L_inf Error':>15} "
    f"{'L2 Error':>15}"
)

print("-" * 54)

for i, N in enumerate(N_values):
    print(
        f"{N:6d} "
        f"{h_values[i]:12.6f} "
        f"{errors_inf[i]:15.6e} "
        f"{errors_2[i]:15.6e}"
    )

print("\nObserved convergence orders:")

for i in range(len(orders_inf)):
    print(
        f"N = {N_values[i]:3d} -> {N_values[i+1]:3d}: "
        f"p_inf = {orders_inf[i]:.4f}, "
        f"p_2 = {orders_2[i]:.4f}"
    )


# ------------------------------------------------------------
# Convergence plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

# Relative L-infinity error
ax.loglog(
    h_values,
    errors_inf,
    "o-",
    color="black",
    linewidth=2,
    markersize=7,
    label=r"Relative error, $p=\infty$"
)

# Relative L2 error
ax.loglog(
    h_values,
    errors_2,
    "s-",
    color="red",
    linewidth=2,
    markersize=7,
    label=r"Relative error, $p=2$"
)

# ------------------------------------------------------------
# Second-order reference curve
# ------------------------------------------------------------

# Shift reference upward for visual clarity
reference = 1.5 * errors_inf[0] * (h_values / h_values[0])**2

ax.loglog(
    h_values,
    reference,
    "^:",
    color="royalblue",
    linewidth=2.5,
    markersize=7,
    label=r"Reference: $\mathcal{O}(\Delta x^2)$"
)

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_xlabel(r"Grid spacing $\Delta x$", fontsize=18)
ax.set_ylabel("Relative Error", fontsize=18)

ax.set_title(
    "Convergence of the Poisson Finite-Difference Solution",
    fontsize=22,
    pad=12
)

ax.grid(
    True,
    which="major",
    linestyle="-",
    alpha=0.25
)

ax.grid(
    True,
    which="minor",
    linestyle=":",
    alpha=0.15
)

ax.legend(
    fontsize=11,
    loc="upper left",
    frameon=True
)

ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()

plt.savefig(
    "P1_Convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()