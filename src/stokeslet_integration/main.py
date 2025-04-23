import time, math
import arguably
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from alive_progress import alive_bar
from prettytable import PrettyTable

from .mesh import validate_partitions_setting, rectangular_mesh, partition
from .stokeslet import (
    integrate_over_elements_wo_singularity,
    integrate_over_element_w_singularity,
    assemble_hydrodynamic_func_matrix,
)


@arguably.command
def __root__(
    *,
    x: list[int],
    y: list[int],
    l: float = 500e-6,
    b: float = 50e-6,
    eps: float = 1e-6,
    f: float = 1e3,
    mu: float = 8.9e-4,
    rho: float = 997,
    n_arc_points: int = 25,
    n_triangles: int = 100,
    factor: float = 1.3,
    mem_priority: bool = False,
    solve_p: bool = False,
    plot: bool = False,
):
    """
    Efficiently computes the hydrodynamic function matrix for a MEMS resonator.

    Args:
        x: [-x/] horizontal {partitions} towards the right edge, e.g., -x 3,5,3 means "create 3 equally sized partitions, refine last partiton into 5 partitions, refine last partition (of those 5 partitions) further into 3 partitions". This results in the following unit lengths for the mesh elements [15, 15, 3, 3, 3, 3, 1, 1, 1]. Single partition setting can also be used, e.g., -x 101, although the size of the hydrodynamic function matrix will be greatly increased. Note: all partition values, except first, must be ODD!
        y: [-y/] vertical {partitions} towards BOTH edges. Same principle as -x setting, except that the setting gets mirrored onto both edges. Note: all partition values, except first, must be ODD!
        l: [-l/] {length} of the resonator plate [m]
        b: [-b/] {width} of the resonator plate [m]
        eps: [-t/] relative error {threshold} for Stokeslet integrals
        f: [-f/] resonator {frequency} [Hz]
        mu: {viscosity} of the surrounding medium [Pa*s]
        rho: {density} of the surrounding medium [kg/m^3]
        n_arc_points: [--arc] # of {points} for defining the arc of the circle segment (more points -> smoother arc)
        n_triangles: [--segment-triangles] {initial} # of triangles for meshing the circle segment
        factor: [--segment-quality] each time integral convergence for a circle segment is not achieved, # of triangles will be multiplied by this {factor}
        mem_priority: if this flag is set, radial symmetry is used during hydrodynamic function matrix assembly by mirroring indices (slower than first expanding to whole padded mesh by flipping + concatenating horizontally and vertically, and using NumPy's fancy indexing on the full padded mesh matrix, but uses less RAM)
        solve_p: [-p/] if this flag is set, v=Sp will be solved for p with v=1
    """
    start_time = time.time()

    validate_partitions_setting(x)
    validate_partitions_setting(y)
    if (len(x) == 1 and len(y) > 1) or (len(y) == 1 and len(x) > 1):
        arguably.error("Can't mix partitioning modes")

    # Step 1: generate mesh elements over which the Stokeslet will be integrated (excl. singularity)
    # Non-dimensionalize l and b
    l_nondim, b_nondim = 1, b / l
    # Stokeslet integral mesh resolution
    I_shape = (
        math.prod(y),
        math.prod(x),
    )  # (rows, cols)
    I_rows, I_cols = I_shape
    # Determine Stokeslet integral mesh unit dimensions
    dx, dy = l_nondim / I_cols, b_nondim / I_rows
    I_elements = rectangular_mesh(I_shape, dx, dy)

    print(f"Unit element aspect ratio: 1:{dx/dy}")

    # Pressure discretization
    p_rows_partitions = None if len(y) == 1 else partition(I_rows, y, "rows")
    p_cols_partitions = None if len(x) == 1 else partition(I_cols, x, "cols")
    p_shape = (
        I_rows if p_rows_partitions is None else len(p_rows_partitions),
        I_cols if p_cols_partitions is None else len(p_cols_partitions),
    )  # (rows, cols)

    table = PrettyTable()
    table.align = "l"
    table.field_names = ["", "Shape (rows, cols)", "Elements"]
    table.add_rows(
        [
            [
                "Stokeslet integrals mesh",
                I_shape,
                f"{math.prod(I_shape):,}",
            ],
            [
                "Discretized pressure mesh",
                p_shape,
                f"{math.prod(p_shape):,}",
            ],
        ],
        divider=True,
    )
    table.add_row(
        [
            "Hydrodynamic function matrix",
            f"{p_shape}^2",
            f"{math.prod(p_shape)**2:,}",
        ],
    )
    print(table)

    # Step 2: integrate over mesh elements excl. singularity
    omega = 2 * np.pi * f
    nu = mu / rho
    lam = np.sqrt(-1 * 1j * omega * l**2 / nu, dtype=np.complex64)
    x_f = np.array([l_nondim - dx / 2, b_nondim - dy / 2, 0])  # Last element
    I = integrate_over_elements_wo_singularity(I_elements, x_f, lam, eps)
    # Step 3: integrate over element with singularity at the center
    I[-1] = integrate_over_element_w_singularity(
        dx, dy, eps, lam, n_arc_points, n_triangles, factor
    )
    I = I.reshape(I_shape)

    if plot:
        plt.figure(figsize=(15, 5))
        plt.title(
            "Stokeslet integrals (x_f in the bottom right corner)",
            fontsize=14,
        )
        fig = plt.pcolormesh(
            np.real(I),
            cmap="magma",
            norm=colors.SymLogNorm(
                linthresh=1e-8, vmin=np.min(np.real(I)), vmax=np.max(np.real(I))
            ),
        )
        plt.gca().invert_yaxis()
        plt.colorbar(fig)
        plt.xlabel("cols")
        plt.ylabel("rows")
        plt.show()

    # Step 4: build hydrodynamic function matrix (S)
    S = assemble_hydrodynamic_func_matrix(
        I, mem_priority, p_shape, p_cols_partitions, p_rows_partitions
    )
    np.savez("S.npz", S=S)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time to compute S: {elapsed_time:.2f} seconds")

    # Step 5: solve for p with v=1
    if solve_p:
        with alive_bar(
            1, title="Solving p", title_length=72, bar=None, stats=False, monitor=False
        ) as bar:
            v = np.ones(p_shape).flatten()
            p_inv_scale = mu / l  # scale back to [Pa]
            p = np.linalg.solve(S, v).reshape(p_shape) * p_inv_scale
            np.savez("p.npz", p=p)
            bar()

        if plot:
            plt.figure(figsize=(15, 8))

            plt.subplot(2, 3, 1)
            plt.title("absolute pressure", fontsize=14)
            plot = plt.pcolormesh(
                np.abs(p),
                antialiased=True,
                linewidth=0.0,
                rasterized=True,
            )
            plt.gca().invert_yaxis()
            plt.colorbar(plot, label="[Pa]")
            plt.xlabel("mesh columns (x-direction)", fontsize=12)
            plt.ylabel("mesh rows (y-direction)", fontsize=12)

            plt.subplot(2, 3, 2)
            plt.title("real pressure", fontsize=14)
            plot = plt.pcolormesh(
                np.real(p),
                antialiased=True,
                linewidth=0.0,
                rasterized=True,
            )
            plt.gca().invert_yaxis()
            plt.colorbar(plot, label="[Pa]")
            plt.xlabel("mesh columns (x-direction)", fontsize=12)
            plt.ylabel("mesh rows (y-direction)", fontsize=12)

            plt.subplot(2, 3, 3)
            plt.title("imaginary pressure", fontsize=14)
            plot = plt.pcolormesh(
                np.abs(np.imag(p)),
                antialiased=True,
                linewidth=0.0,
                rasterized=True,
            )
            plt.gca().invert_yaxis()
            plt.colorbar(plot, label="[Pa]")
            plt.xlabel("mesh columns (x-direction)", fontsize=12)
            plt.ylabel("mesh rows (y-direction)", fontsize=12)

            if p_rows_partitions is not None:
                cumulative_distances = (
                    (
                        np.cumsum(p_rows_partitions, dtype=np.float64)
                        - np.array(p_rows_partitions, dtype=np.float64) / 2
                    )
                    * dy
                    * l
                )
            else:
                cumulative_distances = np.linspace(
                    dy * l / 2, b - (dy * l / 2), p_shape[0]
                )

            plt.subplot(2, 3, 4)
            plt.title("absolute pressure")
            plt.plot(
                cumulative_distances,
                np.abs(p[:, p_shape[1] // 2]),
                ".--",
                linewidth=1,
            )
            plt.xlabel("distance from one edge\n in y-direction [m]", fontsize=12)
            plt.ylabel("[Pa]", fontsize=12)

            plt.subplot(2, 3, 5)
            plt.title("real pressure")
            plt.plot(
                cumulative_distances,
                np.real(p[:, p_shape[1] // 2]),
                ".--",
                linewidth=1,
            )
            plt.xlabel("distance from one edge\n in y-direction [m]", fontsize=12)
            plt.ylabel("[Pa]", fontsize=12)

            plt.subplot(2, 3, 6)
            plt.title("imaginary pressure")
            plt.plot(
                cumulative_distances,
                np.abs(np.imag(p[:, p_shape[1] // 2])),
                ".--",
                linewidth=1,
            )
            plt.xlabel("distance from one edge\n in y-direction [m]", fontsize=12)
            plt.ylabel("[Pa]", fontsize=12)

            plt.tight_layout()
            plt.savefig("p.pdf", format="pdf", bbox_inches="tight", dpi=300)

        # Hydrodynamic force
        unit_element_area = (dx * l) * (dy * l)  # scale back by multiplying with l
        if p_rows_partitions is not None:
            mesh_element_areas = (
                np.array(p_rows_partitions)[:, None]  # shape (rows, 1)
                * np.array(p_cols_partitions)[None, :]  # shape (1, cols)
            ) * unit_element_area  # broadcasts into shape (rows, cols)
        else:
            mesh_element_areas = np.ones(I.shape) * unit_element_area

        F = np.sum(p * mesh_element_areas)  # p -> [Pa], areas -> [m^2] => F -> [N]
        np.savez("F.npz", F=F)
        print(f"Hydrodynamic force: {np.abs(F)} N")


if __name__ == "__main__":
    arguably.run()
