import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from scipy.sparse import diags, eye, kron, coo_matrix
from scipy.sparse.linalg import splu, gmres, LinearOperator


# ============================================================
# MA550 - Homework 1
# Problem 4: Charlie's Rule
#
# Rotating ellipse + pulsating circle
# Moving immersed boundaries
# Warm-started GMRES comparison
# ============================================================

L = 2.0 * np.pi

# Computational parameters
N = 41
N_FRAMES = 36
FPS = 8

# Gaussian regularization
GAUSSIAN_WIDTH = 1.2
KERNEL_RADIUS = 4.0

# GMRES parameters
GMRES_RTOL = 1e-8
GMRES_RESTART = 100
GMRES_MAXITER = 100

# Animation settings
SAVE_GIF = True
SHOW_PLOTS = True

# Custom blue-gray-yellow colormap
cmap = LinearSegmentedColormap.from_list(
    "BlueGrayYellow",
    [
        "#123D77",
        "#777777",
        "#FFE34D"
    ],
    N=256
)


# ============================================================
# 1. Finite-difference Laplacian
# ============================================================

def laplacian_matrix(N, h):

    n = N - 2

    T = diags(
        [
            np.ones(n - 1),
            -2.0 * np.ones(n),
            np.ones(n - 1)
        ],
        [-1, 0, 1],
        format="csr"
    ) / h**2

    I = eye(n, format="csr")

    return (
        kron(I, T, format="csr")
        + kron(T, I, format="csr")
    ).tocsc()


# ============================================================
# 2. Moving immersed boundaries
# ============================================================

def immersed_boundaries(t, h):

    # --------------------------------------------------------
    # Boundary 1: Rotating ellipse
    # --------------------------------------------------------

    a = 0.85
    b = 0.45

    xc1 = 2.15
    yc1 = np.pi

    omega = 2.0 * np.pi * t

    perimeter1 = np.pi * (
        3.0 * (a + b)
        - np.sqrt((3.0*a + b)*(a + 3.0*b))
    )

    N1 = max(16, int(np.ceil(perimeter1 / h)))

    theta1 = np.linspace(
        0.0, 2.0*np.pi, N1, endpoint=False
    )

    x1 = (
        xc1
        + a*np.cos(theta1)*np.cos(omega)
        - b*np.sin(theta1)*np.sin(omega)
    )

    y1 = (
        yc1
        + a*np.cos(theta1)*np.sin(omega)
        + b*np.sin(theta1)*np.cos(omega)
    )

    g1 = 0.8 * np.cos(2.0*theta1)

    # --------------------------------------------------------
    # Boundary 2: Pulsating circle
    # --------------------------------------------------------

    xc2 = 4.35
    yc2 = np.pi

    radius = 0.55 + 0.12*np.sin(2.0*np.pi*t)

    N2 = max(
        16,
        int(np.ceil(2.0*np.pi*0.67 / h))
    )

    theta2 = np.linspace(
        0.0, 2.0*np.pi, N2, endpoint=False
    )

    x2 = xc2 + radius*np.cos(theta2)
    y2 = yc2 + radius*np.sin(theta2)

    g2 = 0.7*np.sin(
        3.0*theta2 + 2.0*np.pi*t
    )

    x_ib = np.concatenate((x1, x2))
    y_ib = np.concatenate((y1, y2))
    g_ib = np.concatenate((g1, g2))

    return {
        "x": x_ib,
        "y": y_ib,
        "g": g_ib,
        "shapes": [
            (x1, y1, g1),
            (x2, y2, g2)
        ]
    }


# ============================================================
# 3. Gaussian interpolation operator
# ============================================================

def build_interpolation_matrix(
    Xi, Yi, x_ib, y_ib, h
):

    a = GAUSSIAN_WIDTH*h
    cutoff = KERNEL_RADIUS*a

    rows = []
    cols = []
    data = []

    n_ib = len(x_ib)

    for k in range(n_ib):

        dx = Xi - x_ib[k]
        dy = Yi - y_ib[k]

        mask = (
            (np.abs(dx) <= cutoff)
            & (np.abs(dy) <= cutoff)
        )

        ids = np.flatnonzero(mask)

        r2 = dx[ids]**2 + dy[ids]**2

        weights = (
            np.exp(-r2/(2.0*a**2))
            / (2.0*np.pi*a**2)
        ) * h**2

        rows.extend([k]*len(ids))
        cols.extend(ids.tolist())
        data.extend(weights.tolist())

    J = coo_matrix(
        (data, (rows, cols)),
        shape=(n_ib, len(Xi))
    ).tocsr()

    return J


# ============================================================
# 4. Reduced Schur-complement solver
# ============================================================

def solve_frame(
    geometry,
    Xi,
    Yi,
    h,
    poisson_lu,
    previous_q=None
):

    x_ib = geometry["x"]
    y_ib = geometry["y"]
    g_ib = geometry["g"]

    J = build_interpolation_matrix(
        Xi, Yi, x_ib, y_ib, h
    )

    # Discrete spreading operator:
    # S = J.T / h^2
    #
    # This scaling corresponds to the Gaussian kernel
    # evaluated on the Cartesian grid.
    S = J.T / h**2

    n_ib = len(g_ib)

    # The PDE is:
    #
    # A u - S q = 0
    #
    # Therefore:
    #
    # u = A^{-1} S q
    #
    # Applying the IB constraint:
    #
    # J A^{-1} S q = g

    def schur_matvec(q):

        spread_q = S @ q

        u = poisson_lu.solve(spread_q)

        return J @ u

    Schur = LinearOperator(
        (n_ib, n_ib),
        matvec=schur_matvec,
        dtype=float
    )

    # --------------------------------------------------------
    # Warm-start GMRES
    # --------------------------------------------------------

    warm_iterations = [0]

    def warm_callback(residual):
        warm_iterations[0] += 1

    start = time.perf_counter()

    q_warm, info_warm = gmres(
        Schur,
        g_ib,
        x0=previous_q,
        rtol=GMRES_RTOL,
        atol=0.0,
        restart=GMRES_RESTART,
        maxiter=GMRES_MAXITER,
        callback=warm_callback,
        callback_type="pr_norm"
    )

    warm_time = time.perf_counter() - start

    # --------------------------------------------------------
    # Cold-start GMRES comparison
    # --------------------------------------------------------

    cold_iterations = [0]

    def cold_callback(residual):
        cold_iterations[0] += 1

    start = time.perf_counter()

    q_cold, info_cold = gmres(
        Schur,
        g_ib,
        x0=np.zeros(n_ib),
        rtol=GMRES_RTOL,
        atol=0.0,
        restart=GMRES_RESTART,
        maxiter=GMRES_MAXITER,
        callback=cold_callback,
        callback_type="pr_norm"
    )

    cold_time = time.perf_counter() - start

    if info_warm != 0 or info_cold != 0:
        raise RuntimeError(
            "GMRES did not converge. "
            f"Warm info={info_warm}, "
            f"cold info={info_cold}"
        )

    # Recover the Cartesian solution
    u = poisson_lu.solve(S @ q_warm)

    # Verify the immersed-boundary condition
    ib_values = J @ u

    ib_error = np.max(
        np.abs(ib_values - g_ib)
    )

    return {
        "u": u,
        "q": q_warm,
        "ib_error": ib_error,
        "warm_iterations": warm_iterations[0],
        "cold_iterations": cold_iterations[0],
        "warm_time": warm_time,
        "cold_time": cold_time
    }


# ============================================================
# 5. Set up Cartesian grid
# ============================================================

x = np.linspace(0.0, L, N)
y = np.linspace(0.0, L, N)

h = x[1] - x[0]

X, Y = np.meshgrid(
    x, y, indexing="ij"
)

Xi = X[1:-1, 1:-1].ravel()
Yi = Y[1:-1, 1:-1].ravel()

A = laplacian_matrix(N, h)

# Factorize the Poisson matrix once.
# It remains unchanged as the boundaries move.
poisson_lu = splu(A)


# ============================================================
# 6. Solve all animation frames
# ============================================================

solutions = []
geometries = []

warm_times = []
cold_times = []

warm_iterations = []
cold_iterations = []

ib_errors = []

previous_q = None

print("="*70)
print("PROBLEM 4: CHARLIE'S RULE")
print("="*70)

print(f"Grid size: {N} x {N}")
print(f"Grid spacing: {h:.6f}")
print(f"Animation frames: {N_FRAMES}")
print()

for frame in range(N_FRAMES):

    t = frame/N_FRAMES

    geometry = immersed_boundaries(t, h)

    result = solve_frame(
        geometry,
        Xi,
        Yi,
        h,
        poisson_lu,
        previous_q
    )

    U = np.zeros_like(X)

    U[1:-1, 1:-1] = result["u"].reshape(
        (N-2, N-2)
    )

    solutions.append(U)
    geometries.append(geometry)

    warm_times.append(result["warm_time"])
    cold_times.append(result["cold_time"])

    warm_iterations.append(
        result["warm_iterations"]
    )

    cold_iterations.append(
        result["cold_iterations"]
    )

    ib_errors.append(result["ib_error"])

    previous_q = result["q"].copy()

    print(
        f"Frame {frame+1:02d}/{N_FRAMES} | "
        f"IB error = {result['ib_error']:.2e} | "
        f"Warm = {result['warm_iterations']:4d} | "
        f"Cold = {result['cold_iterations']:4d}"
    )


# ============================================================
# 7. 2D animation
# ============================================================

fig2d, ax2d = plt.subplots(figsize=(8, 7))

vmax = max(
    np.max(np.abs(U)) for U in solutions
)

def update_2d(frame):

    ax2d.clear()

    U = solutions[frame]
    geometry = geometries[frame]

    ax2d.contourf(
        X, Y, U,
        levels=np.linspace(-vmax, vmax, 60),
        cmap=cmap,
        extend="both"
    )

    for xs, ys, gs in geometry["shapes"]:

        ax2d.plot(
            np.r_[xs, xs[0]],
            np.r_[ys, ys[0]],
            "k-",
            linewidth=2.5
        )

        ax2d.plot(
            xs, ys,
            "ko",
            markersize=2
        )

    ax2d.set_xlim(0, L)
    ax2d.set_ylim(0, L)

    ax2d.set_aspect("equal")

    ax2d.set_xlabel(r"$x$")
    ax2d.set_ylabel(r"$y$")

    ax2d.set_title(
        f"Moving Immersed Boundaries — Frame {frame+1}"
    )

    return []


animation2d = FuncAnimation(
    fig2d,
    update_2d,
    frames=N_FRAMES,
    interval=1000/FPS
)

if SAVE_GIF:

    print("\nSaving 2D animation...")

    animation2d.save(
        "P4_2D_Animation.gif",
        writer=PillowWriter(fps=FPS)
    )


# ============================================================
# 8. 3D animation
# ============================================================

fig3d = plt.figure(figsize=(11, 8))

ax3d = fig3d.add_subplot(
    111,
    projection="3d",
    computed_zorder=False
)

def update_3d(frame):

    ax3d.clear()

    U = solutions[frame]
    geometry = geometries[frame]

    ax3d.plot_surface(
        X, Y, U,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        alpha=0.90,
        edgecolor="none",
        rcount=60,
        ccount=60,
        zorder=1
    )

    for xs, ys, gs in geometry["shapes"]:

        ax3d.plot(
            np.r_[xs, xs[0]],
            np.r_[ys, ys[0]],
            np.r_[gs, gs[0]],
            color="black",
            linewidth=3,
            zorder=100
        )

        ax3d.scatter(
            xs, ys, gs,
            color="black",
            s=8,
            depthshade=False,
            zorder=101
        )

    ax3d.set_xlim(0, L)
    ax3d.set_ylim(0, L)

    ax3d.set_zlim(-1.2*vmax, 1.2*vmax)

    ax3d.set_xlabel(r"$x$")
    ax3d.set_ylabel(r"$y$")
    ax3d.set_zlabel(r"$u(x,y)$")

    ax3d.set_title(
        f"Charlie’s Rule: Deforming Surface — Frame {frame+1}"
    )

    ax3d.view_init(
        elev=23,
        azim=-125
    )

    ax3d.set_box_aspect((1.5, 1.5, 0.7))

    return []


animation3d = FuncAnimation(
    fig3d,
    update_3d,
    frames=N_FRAMES,
    interval=1000/FPS
)

if SAVE_GIF:

    print("Saving 3D animation...")

    animation3d.save(
        "P4_3D_Animation.gif",
        writer=PillowWriter(fps=FPS)
    )


# ============================================================
# 9. GMRES performance comparison
# ============================================================

frames = np.arange(1, N_FRAMES+1)

fig, axes = plt.subplots(
    2, 1,
    figsize=(9, 9),
    sharex=True
)

axes[0].plot(
    frames,
    cold_iterations,
    "k-o",
    label="Cold start"
)

axes[0].plot(
    frames,
    warm_iterations,
    "r-s",
    label="Warm start"
)

axes[0].set_ylabel("GMRES iterations")
axes[0].set_title(
    "GMRES Performance: Warm vs Cold Start"
)

axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(
    frames,
    cold_times,
    "k-o",
    label="Cold start"
)

axes[1].plot(
    frames,
    warm_times,
    "r-s",
    label="Warm start"
)

axes[1].set_xlabel("Animation frame")
axes[1].set_ylabel("Solve time (seconds)")

axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()

plt.savefig(
    "P4_GMRES_Performance.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 10. Summary
# ============================================================

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

print(
    f"Average cold-start iterations: "
    f"{np.mean(cold_iterations):.2f}"
)

print(
    f"Average warm-start iterations: "
    f"{np.mean(warm_iterations):.2f}"
)

print(
    f"Average cold-start time: "
    f"{np.mean(cold_times):.4f} seconds"
)

print(
    f"Average warm-start time: "
    f"{np.mean(warm_times):.4f} seconds"
)

print(
    f"Maximum IB constraint error: "
    f"{np.max(ib_errors):.6e}"
)

print("\nSaved files:")

print("  P4_2D_Animation.gif")
print("  P4_3D_Animation.gif")
print("  P4_GMRES_Performance.png")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")