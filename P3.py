import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial import cKDTree


# ============================================================
# MA550 - Homework 1
# Problem 3: An Almost Minimal Surface
# ============================================================

L = 2 * np.pi


# ------------------------------------------------------------
# Immersed boundary Dirichlet data
# ------------------------------------------------------------

def boundary_function(x, y):
    return np.cos(2 * x) * np.sin(2 * y)


# ------------------------------------------------------------
# Immersed boundary geometry
# ------------------------------------------------------------

def immersed_boundary(N_ib):

    theta = np.linspace(
        0, 2 * np.pi, N_ib, endpoint=False
    )

    x_ib = np.pi + np.cos(theta)
    y_ib = np.pi + np.sin(theta)

    return x_ib, y_ib


# ------------------------------------------------------------
# Regularized Gaussian delta
# ------------------------------------------------------------

def gaussian_delta(dx, dy, a):

    r2 = dx**2 + dy**2

    return (
        np.exp(-r2 / (2 * a**2))
        / (2 * np.pi * a**2)
    )


# ------------------------------------------------------------
# Finite difference Laplacian
# ------------------------------------------------------------

def laplacian_matrix(N, h):

    n = N - 2

    T = diags(
        [
            np.ones(n - 1),
            -2 * np.ones(n),
            np.ones(n - 1)
        ],
        offsets=[-1, 0, 1],
        format="csr"
    ) / h**2

    I = eye(n, format="csr")

    A = (
        kron(I, T, format="csr")
        + kron(T, I, format="csr")
    )

    return A


# ------------------------------------------------------------
# Localized immersed boundary weights
# ------------------------------------------------------------

def build_ib_weights(X, Y, x_ib, y_ib, h, a):

    Xi = X[1:-1, 1:-1].ravel()
    Yi = Y[1:-1, 1:-1].ravel()

    grid_points = np.column_stack((Xi, Yi))
    ib_points = np.column_stack((x_ib, y_ib))

    tree = cKDTree(grid_points)

    radius = 4 * a

    neighbors = tree.query_ball_point(
        ib_points, radius
    )

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
# Interpolation operator
# ------------------------------------------------------------

def interpolate(u, indices, weights):

    result = np.zeros(len(indices))

    for k, (ids, w) in enumerate(
        zip(indices, weights)
    ):

        result[k] = np.dot(w, u[ids])

    return result


# ------------------------------------------------------------
# Spreading operator
# ------------------------------------------------------------

def spread(q, indices, weights, n_unknowns):

    result = np.zeros(n_unknowns)

    for k, (ids, w) in enumerate(
        zip(indices, weights)
    ):

        np.add.at(
            result,
            ids,
            q[k] * w
        )

    return result


# ============================================================
# Immersed boundary Poisson solver
# ============================================================

def poisson_ib_solver(N, spacing_factor=1.0, rtol=1e-8):

    start = time.perf_counter()

    # --------------------------------------------------------
    # Cartesian grid
    # --------------------------------------------------------

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    h = x[1] - x[0]

    X, Y = np.meshgrid(
        x, y, indexing="ij"
    )

    n = N - 2
    n_unknowns = n**2

    # --------------------------------------------------------
    # Immersed boundary
    # --------------------------------------------------------

    circumference = 2 * np.pi

    N_ib = max(
        8,
        int(np.ceil(
            circumference / (spacing_factor * h)
        ))
    )

    x_ib, y_ib = immersed_boundary(N_ib)

    g_ib = boundary_function(x_ib, y_ib)

    # Gaussian width
    a = 1.2 * h

    # --------------------------------------------------------
    # Discrete Laplacian
    # --------------------------------------------------------

    A = laplacian_matrix(N, h)

    # --------------------------------------------------------
    # Homogeneous Dirichlet boundary conditions
    #
    # f(x,y) = 0
    # u = 0 on the outer boundary
    #
    # Therefore the entire PDE RHS is zero.
    # --------------------------------------------------------

    b = np.zeros(n_unknowns)

    # --------------------------------------------------------
    # Immersed boundary operators
    # --------------------------------------------------------

    indices, weights = build_ib_weights(
        X, Y, x_ib, y_ib, h, a
    )

    # --------------------------------------------------------
    # Matrix-free saddle-point system
    #
    # [ A  -S ] [u] = [0]
    # [ J   0 ] [q]   [g]
    # --------------------------------------------------------

    def matvec(z):

        u = z[:n_unknowns]
        q = z[n_unknowns:]

        upper = (
            A @ u
            - spread(
                q,
                indices,
                weights,
                n_unknowns
            )
        )

        lower = interpolate(
            u, indices, weights
        )

        return np.concatenate(
            (upper, lower)
        )

    size = n_unknowns + N_ib

    Amult = LinearOperator(
        (size, size),
        matvec=matvec,
        dtype=float
    )

    rhs = np.concatenate(
        (b, g_ib)
    )

    # --------------------------------------------------------
    # GMRES solver
    # --------------------------------------------------------

    iterations = [0]

    def callback(residual):
        iterations[0] += 1

    solution, info = gmres(
        Amult,
        rhs,
        restart=100,
        rtol=1e-6,
        atol=0.0,
        maxiter=100,
        callback=callback,
        callback_type="pr_norm"
)

    if info != 0:
        print(
            f"WARNING: GMRES did not fully converge "
            f"(info={info})"
        )

    # --------------------------------------------------------
    # Reconstruct numerical solution
    # --------------------------------------------------------

    u_interior = solution[
        :n_unknowns
    ].reshape((n, n))

    q = solution[n_unknowns:]

    # Outer boundaries remain zero
    U = np.zeros_like(X)

    U[1:-1, 1:-1] = u_interior

    # --------------------------------------------------------
    # Interpolated immersed boundary values
    # --------------------------------------------------------

    u_ib = interpolate(
        solution[:n_unknowns],
        indices,
        weights
    )

    ib_error = np.abs(u_ib - g_ib)

    ib_residual = np.max(ib_error)

    # Full saddle-point residual
    residual = Amult @ solution - rhs

    relative_residual = (
        np.linalg.norm(residual)
        / np.linalg.norm(rhs)
    )

    elapsed = time.perf_counter() - start

    return {
        "X": X,
        "Y": Y,
        "U": U,
        "x_ib": x_ib,
        "y_ib": y_ib,
        "g_ib": g_ib,
        "u_ib": u_ib,
        "q": q,
        "h": h,
        "N_ib": N_ib,
        "ib_residual": ib_residual,
        "relative_residual": relative_residual,
        "iterations": iterations[0],
        "time": elapsed,
        "info": info
    }


# ============================================================
# Part 1: Solve the PDE
# ============================================================

N = 61

result = poisson_ib_solver(N)

X = result["X"]
Y = result["Y"]
U = result["U"]

x_ib = result["x_ib"]
y_ib = result["y_ib"]

g_ib = result["g_ib"]
u_ib = result["u_ib"]

print("=" * 65)
print("PROBLEM 3: AN ALMOST MINIMAL SURFACE")
print("=" * 65)

print(f"Grid points per direction: {N}")
print(f"Grid spacing: {result['h']:.6f}")
print(f"Immersed boundary points: {result['N_ib']}")

print(f"Maximum solution: {np.max(U):.6e}")
print(f"Minimum solution: {np.min(U):.6e}")

print(f"IB constraint residual: {result['ib_residual']:.6e}")
print(f"Relative system residual: {result['relative_residual']:.6e}")

print(f"GMRES iterations: {result['iterations']}")
print(f"Elapsed time: {result['time']:.3f} seconds")


# ============================================================
# Part 2: 2D visualization
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

contour = ax.contourf(
    X, Y, U,
    levels=60,
    cmap="cividis"
)

fig.colorbar(
    contour,
    ax=ax,
    label=r"$u(x,y)$"
)

# Close immersed boundary
x_closed = np.r_[x_ib, x_ib[0]]
y_closed = np.r_[y_ib, y_ib[0]]

ax.plot(
    x_closed,
    y_closed,
    "k-",
    linewidth=2.5,
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

ax.set_title(
    "Almost Minimal Surface: Top View"
)

ax.legend()

plt.tight_layout()
plt.savefig(
    "P3_2D.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# Part 3: 3D visualization
# ============================================================

fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(
    111,
    projection="3d",
    computed_zorder=False
)

surf = ax.plot_surface(
    X, Y, U,
    cmap="turbo",
    edgecolor="none",
    alpha=0.90,
    antialiased=True,
    rcount=100,
    ccount=100,
    zorder=1
)

# Immersed boundary follows prescribed boundary values
z_ib = g_ib

z_closed = np.r_[z_ib, z_ib[0]]

line, = ax.plot(
    x_closed,
    y_closed,
    z_closed,
    color="black",
    linewidth=3,
    zorder=100,
    label="Immersed Boundary"
)

ax.scatter(
    x_ib,
    y_ib,
    z_ib,
    color="black",
    s=12,
    depthshade=False,
    zorder=101
)

line.set_path_effects([
    pe.Stroke(
        linewidth=4,
        foreground="white"
    ),
    pe.Normal()
])

ax.set_xlabel(r"$x$", fontsize=12, labelpad=10)
ax.set_ylabel(r"$y$", fontsize=12, labelpad=10)
ax.set_zlabel(r"$u(x,y)$", fontsize=12, labelpad=8)

ax.set_title(
    "Almost Minimal Surface with Immersed Boundary",
    fontsize=15,
    pad=15
)

ax.view_init(elev=20, azim=-120)

ax.set_box_aspect((1.5, 1.5, 0.6))

ax.set_xlim(0, L)
ax.set_ylim(0, L)

ax.legend(loc="upper right")

fig.colorbar(
    surf,
    ax=ax,
    shrink=0.65,
    pad=0.1,
    label=r"$u(x,y)$"
)

plt.tight_layout()
plt.savefig(
    "P3_3D.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# Part 4: Immersed boundary verification
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

theta = np.linspace(
    0, 2 * np.pi,
    result["N_ib"],
    endpoint=False
)

ax.plot(
    theta,
    g_ib,
    "k-",
    linewidth=2,
    label="Prescribed IB values"
)

ax.plot(
    theta,
    u_ib,
    "ro",
    markersize=3,
    label="Interpolated numerical values"
)

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$u(x^{ib},y^{ib})$")

ax.set_title(
    "Verification of Immersed Boundary Conditions"
)

ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()
plt.savefig(
    "P3_IB_Verification.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()