import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial import cKDTree


# ============================================================
# MA550 - Homework 2
# Problem 2: Poisson Equation with an Immersed Boundary
# ============================================================

L = 2 * np.pi


# ------------------------------------------------------------
# Manufactured solution and forcing
# ------------------------------------------------------------

def u_exact(x, y):
    return np.cos(2 * x) * np.sin(2 * y)


def forcing(x, y):
    return -8 * u_exact(x, y)


# ------------------------------------------------------------
# Immersed boundary geometry
# ------------------------------------------------------------

def immersed_boundary(N_ib):

    theta = np.linspace(0, 2 * np.pi, N_ib, endpoint=False)

    x_ib = np.pi + np.cos(theta)
    y_ib = np.pi + np.sin(theta)

    return x_ib, y_ib


# ------------------------------------------------------------
# Regularized Gaussian delta
#
# delta_a(x,y) = 1/(2*pi*a^2)
#                exp(-(x^2+y^2)/(2*a^2))
#
# Here a = 1.2*h is interpreted as the Gaussian width.
# ------------------------------------------------------------

def gaussian_delta(dx, dy, a):

    r2 = dx**2 + dy**2

    return np.exp(-r2 / (2 * a**2)) / (2 * np.pi * a**2)


# ------------------------------------------------------------
# Construct finite difference Laplacian
# ------------------------------------------------------------

def laplacian_matrix(N, h):

    n = N - 2

    T = diags(
        [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)],
        [-1, 0, 1],
        format="csr"
    ) / h**2

    I = eye(n, format="csr")

    return kron(I, T, format="csr") + kron(T, I, format="csr")


# ------------------------------------------------------------
# Construct localized interpolation/spreading weights
#
# Only points within 4 Gaussian widths are retained.
# This avoids constructing a dense matrix.
# ------------------------------------------------------------

def build_ib_weights(X, Y, x_ib, y_ib, h, a):

    n = X.shape[0] - 2

    Xi = X[1:-1, 1:-1].ravel()
    Yi = Y[1:-1, 1:-1].ravel()

    grid_points = np.column_stack((Xi, Yi))
    ib_points = np.column_stack((x_ib, y_ib))

    tree = cKDTree(grid_points)

    radius = 4 * a

    neighbors = tree.query_ball_point(ib_points, radius)

    indices = []
    weights = []

    for k, ids in enumerate(neighbors):

        ids = np.asarray(ids, dtype=int)

        dx = Xi[ids] - x_ib[k]
        dy = Yi[ids] - y_ib[k]

        w = gaussian_delta(dx, dy, a) * h**2

        indices.append(ids)
        weights.append(w)

    return indices, weights


# ------------------------------------------------------------
# Interpolation operator J
# ------------------------------------------------------------

def interpolate(u, indices, weights):

    result = np.zeros(len(indices))

    for k, (ids, w) in enumerate(zip(indices, weights)):
        result[k] = np.dot(w, u[ids])

    return result


# ------------------------------------------------------------
# Spreading operator S
# ------------------------------------------------------------

def spread(q, indices, weights, n_unknowns):

    result = np.zeros(n_unknowns)

    for k, (ids, w) in enumerate(zip(indices, weights)):
        np.add.at(result, ids, q[k] * w)

    return result


# ============================================================
# Immersed boundary Poisson solver
# ============================================================

def poisson_ib_solver(N, spacing_factor=1.0, rtol=1e-7):

    start = time.perf_counter()

    # Cartesian grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    U_exact = u_exact(X, Y)

    n = N - 2
    n_unknowns = n**2

    # --------------------------------------------------------
    # Immersed boundary points
    # --------------------------------------------------------

    circumference = 2 * np.pi

    N_ib = max(
        8,
        int(np.ceil(circumference / (spacing_factor * h)))
    )

    x_ib, y_ib = immersed_boundary(N_ib)

    g_ib = u_exact(x_ib, y_ib)

    # Gaussian width
    a = 1.2 * h

    # --------------------------------------------------------
    # Discrete Laplacian
    # --------------------------------------------------------

    A = laplacian_matrix(N, h)

    # --------------------------------------------------------
    # Right-hand side, including outer Dirichlet conditions
    # --------------------------------------------------------

    F = forcing(X, Y)

    b = F[1:-1, 1:-1].copy()

    b[0, :] -= U_exact[0, 1:-1] / h**2
    b[-1, :] -= U_exact[-1, 1:-1] / h**2

    b[:, 0] -= U_exact[1:-1, 0] / h**2
    b[:, -1] -= U_exact[1:-1, -1] / h**2

    b = b.ravel()

    # --------------------------------------------------------
    # Immersed boundary interpolation/spreading weights
    # --------------------------------------------------------

    indices, weights = build_ib_weights(
        X, Y, x_ib, y_ib, h, a
    )

    # --------------------------------------------------------
    # Matrix-free saddle-point operator
    #
    # [ A  -S ] [u] = [f]
    # [ J   0 ] [q]   [g]
    # --------------------------------------------------------

    def matvec(z):

        u = z[:n_unknowns]
        q = z[n_unknowns:]

        upper = A @ u - spread(
            q, indices, weights, n_unknowns
        )

        lower = interpolate(u, indices, weights)

        return np.concatenate((upper, lower))

    size = n_unknowns + N_ib

    Amult = LinearOperator(
        (size, size),
        matvec=matvec,
        dtype=float
    )

    # --------------------------------------------------------
    # Solve using GMRES
    # --------------------------------------------------------

    rhs = np.concatenate((b, g_ib))

    iterations = [0]

    def callback(residual):
        iterations[0] += 1

    solution, info = gmres(
        Amult,
        rhs,
        restart=1000,
        rtol=rtol,
        atol=0.0,
        maxiter=1000,
        callback=callback,
        callback_type="pr_norm"
    )

    if info != 0:
        print(f"WARNING: GMRES did not fully converge (info={info})")

    # --------------------------------------------------------
    # Reconstruct full solution
    # --------------------------------------------------------

    u_interior = solution[:n_unknowns].reshape((n, n))
    q = solution[n_unknowns:]

    U = U_exact.copy()
    U[1:-1, 1:-1] = u_interior

    elapsed = time.perf_counter() - start

    # Relative errors
    E = U - U_exact

    error_inf = (
        np.max(np.abs(E))
        / np.max(np.abs(U_exact))
    )

    error_2 = (
        np.linalg.norm(E.ravel())
        / np.linalg.norm(U_exact.ravel())
    )

    # Immersed boundary constraint residual
    ib_residual = np.max(
        np.abs(interpolate(solution[:n_unknowns], indices, weights) - g_ib)
    )

    return {
        "X": X,
        "Y": Y,
        "U": U,
        "U_exact": U_exact,
        "x_ib": x_ib,
        "y_ib": y_ib,
        "q": q,
        "h": h,
        "N_ib": N_ib,
        "error_inf": error_inf,
        "error_2": error_2,
        "ib_residual": ib_residual,
        "iterations": iterations[0],
        "time": elapsed,
        "info": info
    }


# ============================================================
# Part 1: Numerical solution
# ============================================================

N = 41

result = poisson_ib_solver(N)

X = result["X"]
Y = result["Y"]
U = result["U"]

x_ib = result["x_ib"]
y_ib = result["y_ib"]

print("=" * 65)
print("POISSON EQUATION WITH IMMERSED BOUNDARY")
print("=" * 65)

print(f"Grid points per direction: {N}")
print(f"Grid spacing: {result['h']:.6f}")
print(f"Immersed boundary points: {result['N_ib']}")
print(f"Relative L_inf error: {result['error_inf']:.6e}")
print(f"Relative L2 error: {result['error_2']:.6e}")
print(f"IB constraint residual: {result['ib_residual']:.6e}")
print(f"GMRES iterations: {result['iterations']}")
print(f"Elapsed time: {result['time']:.3f} seconds")


# ------------------------------------------------------------
# 2D visualization
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))

contour = ax.contourf(
    X, Y, U,
    levels=60,
    cmap="cividis"
)

fig.colorbar(contour, ax=ax, label=r"$u(x,y)$")

ax.plot(
    np.r_[x_ib, x_ib[0]],
    np.r_[y_ib, y_ib[0]],
    "k-",
    linewidth=2,
    label="Immersed Boundary"
)

ax.plot(
    x_ib,
    y_ib,
    "ko",
    markersize=2
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_aspect("equal")
ax.set_title("Poisson Equation with an Immersed Boundary")
ax.legend()

plt.tight_layout()
plt.savefig("P2_2D.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# 3D visualization with clearly visible immersed boundary
# ------------------------------------------------------------

import matplotlib.patheffects as pe

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

# Numerical solution surface
surf = ax.plot_surface(
    X, Y, U,
    cmap="cividis",
    edgecolor="none",
    alpha=0.65,
    antialiased=True,
    rcount=100,
    ccount=100,
    zorder=1
)

# Immersed boundary coordinates
z_ib = u_exact(x_ib, y_ib)

# Small vertical offset for visualization
offset = 0.15
z_ib_visible = z_ib + offset

# Close the immersed boundary curve
x_closed = np.r_[x_ib, x_ib[0]]
y_closed = np.r_[y_ib, y_ib[0]]
z_closed = np.r_[z_ib_visible, z_ib_visible[0]]

# Plot immersed boundary above the surface
line, = ax.plot(
    x_closed,
    y_closed,
    z_closed,
    color="black",
    linewidth=4,
    zorder=100,
    label="Immersed Boundary"
)

# White outline makes the curve visible against dark regions
line.set_path_effects([
    pe.Stroke(linewidth=6, foreground="white"),
    pe.Normal()
])

# Immersed boundary markers
ax.scatter(
    x_ib,
    y_ib,
    z_ib_visible,
    color="black",
    s=25,
    depthshade=False,
    zorder=101
)

# Colorbar
fig.colorbar(
    surf,
    ax=ax,
    shrink=0.65,
    pad=0.1,
    label=r"$u(x,y)$"
)

# Labels
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u(x,y)$")

ax.set_title("Poisson Solution with Immersed Boundary")

# Viewing angle
ax.view_init(elev=35, azim=-60)

ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("P2_3D.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Part 2: Immersed boundary spacing experiment
# ============================================================

spacing_factors = [0.5, 1.0, 2.0, 4.0]

spacing_results = []

print("\n" + "=" * 65)
print("IMMERSED BOUNDARY SPACING EXPERIMENT")
print("=" * 65)

print(
    f"{'Factor':>8} "
    f"{'N_ib':>8} "
    f"{'L_inf Error':>15} "
    f"{'Iterations':>12} "
    f"{'Time (s)':>12}"
)

for factor in spacing_factors:

    r = poisson_ib_solver(N, spacing_factor=factor)

    spacing_results.append(r)

    print(
        f"{factor:8.2f} "
        f"{r['N_ib']:8d} "
        f"{r['error_inf']:15.6e} "
        f"{r['iterations']:12d} "
        f"{r['time']:12.3f}"
    )


# ------------------------------------------------------------
# Accuracy versus IB spacing
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(
    spacing_factors,
    [r["error_inf"] for r in spacing_results],
    "o-",
    color="black",
    linewidth=2,
    label=r"Relative $L_\infty$ error"
)

ax.loglog(
    spacing_factors,
    [r["error_2"] for r in spacing_results],
    "s-",
    color="red",
    linewidth=2,
    label=r"Relative $L_2$ error"
)

ax.set_xlabel(r"IB spacing / $\Delta x$")
ax.set_ylabel("Relative Error")
ax.set_title("Accuracy versus Immersed Boundary Spacing")

ax.grid(True, which="both", alpha=0.25)
ax.legend()

plt.tight_layout()
plt.savefig("P2_Spacing_Accuracy.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# Runtime versus IB spacing
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    spacing_factors,
    [r["time"] for r in spacing_results],
    "o-",
    color="black",
    linewidth=2
)

ax.set_xlabel(r"IB spacing / $\Delta x$")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Computational Cost versus IB Spacing")

ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("P2_Spacing_Runtime.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Part 3: Grid refinement and convergence study
# ============================================================

N_values = [21, 31, 41, 61, 81]

h_values = []
errors_inf = []
errors_2 = []

print("\n" + "=" * 65)
print("GRID REFINEMENT STUDY")
print("=" * 65)

print(
    f"{'N':>6} "
    f"{'Delta x':>12} "
    f"{'L_inf Error':>15} "
    f"{'L2 Error':>15} "
    f"{'Iterations':>12}"
)

for N_grid in N_values:

    r = poisson_ib_solver(N_grid)

    h_values.append(r["h"])
    errors_inf.append(r["error_inf"])
    errors_2.append(r["error_2"])

    print(
        f"{N_grid:6d} "
        f"{r['h']:12.6f} "
        f"{r['error_inf']:15.6e} "
        f"{r['error_2']:15.6e} "
        f"{r['iterations']:12d}"
    )

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

ax.set_xlabel(r"Grid spacing $\Delta x$", fontsize=12)
ax.set_ylabel("Relative Error", fontsize=12)

ax.set_title(
    "Convergence of the Immersed Boundary Poisson Solver",
    fontsize=14,
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
    "P2_Convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()