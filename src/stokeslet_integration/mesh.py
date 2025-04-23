import numpy as np
from meshpy.triangle import MeshInfo, build
from typing import Literal
from arguably import error


def validate_partitions_setting(partitions: list[int]):
    """Input validation

    Args:
        partitions: x or y partitions
    """
    for i, partition in enumerate(partitions):
        if i == 0:
            continue
        if partition % 2 == 0:
            error("All partitions must be odd")
        if partition == 1:
            error("Partitions can't be 1")


def rectangular_mesh(shape: tuple[int, int], dx: float, dy: float):
    """Generates rectangular mesh elements in row-major order over which the Stokeslet will be numerically integrated

    Skips last element where the field point x_f will be set -> singularity)

    Args:
        shape: tuple (rows, cols)
        dx: mesh element length (non-dim)
        dy: mesh element width (non-dim)

    Returns:
        elements (np.ndarray): rectangular mesh elements in `quadpy` shape `(2, 2, n_elements - 1, 2)`
    """
    rows, cols = shape
    n_elements = rows * cols
    elements = np.empty((2, 2, n_elements - 1, 2), dtype=np.float64)
    # Generate rectangular mesh elements
    ij = 0
    for i in range(rows):
        for j in range(cols):
            if ij + 1 == n_elements:
                continue  # Skip (last) singularity element
            # Corner coordinates
            x0, x1 = j * dx, (j + 1) * dx
            y0, y1 = i * dy, (i + 1) * dy
            # Lower-left
            elements[0, 0, ij, 0] = x0
            elements[0, 0, ij, 1] = y0
            # Lower-right
            elements[0, 1, ij, 0] = x1
            elements[0, 1, ij, 1] = y0
            # Upper-left
            elements[1, 0, ij, 0] = x0
            elements[1, 0, ij, 1] = y1
            # Upper-right
            elements[1, 1, ij, 0] = x1
            elements[1, 1, ij, 1] = y1
            ij += 1

    return elements


def partition(n_elements: int, partitions: list[int], axis: Literal["rows", "cols"]):
    """Creates horizontal partitions (only towards right edge), or vertical partitions (applies same setting to both edges).


    Args:
        n_elements: number of unit elements
        partitions: partition setting
        axis: "rows" | "cols"

    Returns:
        result: list of partitions, `sum(result) == n_elements`
    """
    result = []
    parent_partition_size = n_elements
    for i, n in enumerate(partitions):
        partition_size = parent_partition_size // n
        parent_partition_size = partition_size

        if axis == "rows" and i == 0:
            result.extend([partition_size] * (n - 2))
            continue

        if i != len(partitions) - 1:
            result.extend([partition_size] * (n - 1))
        else:
            result.extend([partition_size] * n)

    if axis == "rows":
        result = [*result[partitions[0] - 2 :][::-1], *result]

    return result


def subdivide_element(element: np.ndarray):
    """Subdivides mesh element into 4 same-sized elements

    Args:
        element: mesh element in `quadpy` shape `(2, 2, 1, 2)`

    Returns:
        elements: list of four same-sized mesh elements each in `quadpy` shape `(2, 2, 1, 2)` in the order bottom left, bottom right, top left, top right
    """
    # Corners
    ll_corner = element[0, 0, 0]
    lr_corner = element[0, 1, 0]
    ul_corner = element[1, 0, 0]
    ur_corner = element[1, 1, 0]
    # Midpoints of edges
    b_mid_point = (ll_corner + lr_corner) / 2
    l_mid_point = (ll_corner + ul_corner) / 2
    r_mid_point = (lr_corner + ur_corner) / 2
    t_mid_point = (ul_corner + ur_corner) / 2
    # Centroid
    centroid = (ll_corner + lr_corner + ul_corner + ur_corner) / 4
    # New elements
    bl_element = np.array([[ll_corner, b_mid_point], [l_mid_point, centroid]]).reshape(
        2, 2, 1, 2
    )
    br_element = np.array([[b_mid_point, lr_corner], [centroid, r_mid_point]]).reshape(
        2, 2, 1, 2
    )
    tl_element = np.array([[l_mid_point, centroid], [ul_corner, t_mid_point]]).reshape(
        2, 2, 1, 2
    )
    tr_element = np.array([[centroid, r_mid_point], [t_mid_point, ur_corner]]).reshape(
        2, 2, 1, 2
    )
    return [bl_element, br_element, tl_element, tr_element]


def circle_segment_mesh(
    type: str,
    dx: float,
    dy: float,
    r: float,
    n_arc_points: int,
    n_triangles: int,
):
    """Generates a triangle mesh for a circle segment using MeshPy

    Args:
        type: "top" | "left"
        dx: mesh element length (non-dim)
        dy: mesh element width (non-dim)
        r: radius
        n_arc_points: number of points for defining the arc (more points -> smoother arc)
        n_triangles: number of triangles

    Returns:
        mesh (MeshInfo): `MeshPy` mesh object
    """
    # Generate arc points
    arc_points = []
    if type == "top":
        theta_start = np.arctan2(dy / 2, -dx / 2)
        theta_end = np.arctan2(dy / 2, dx / 2)
    if type == "left":
        theta_start = np.arctan2(dy / 2, -dx / 2)
        theta_end = np.arctan2(-dy / 2, -dx / 2) + 2 * np.pi
    for i in range(n_arc_points):
        theta = theta_start + (theta_end - theta_start) * i / n_arc_points
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        arc_points.append((x, y))

    # Close boundary
    if type == "top":
        boundary = np.array(arc_points + [(dx / 2, dy / 2)])
    if type == "left":
        boundary = np.array(arc_points + [(-dx / 2, -dy / 2)])

    # Connect arc points
    facets = [(i, i + 1) for i in range(len(boundary) - 1)]
    facets.append((len(boundary) - 1, 0))

    # Compute max_volume
    if type == "top":
        theta_central = 2 * np.arcsin(dx / (2 * r))
    if type == "left":
        theta_central = 2 * np.arcsin(dy / (2 * r))
    A = 1 / 2 * (r**2) * (theta_central - np.sin(theta_central))
    max_volume = A / n_triangles

    # Build mesh
    mesh_info = MeshInfo()
    mesh_info.set_points(boundary)
    mesh_info.set_facets(facets)
    segment_mesh = build(mesh_info, max_volume=max_volume, quality_meshing=True)

    return segment_mesh


def reshape_meshpy_for_quadpy(mesh):
    """Reshapes `MeshPy` mesh object into correct shape for `quadpy`

    Args:
        mesh: `MeshPy` mesh object

    Returns:
        elements: triangles in `quadpy` shape `(3, n_elements, 2)`
    """
    triangles = np.array(mesh.elements)  # Shape: (n_triangles, 3)
    points = np.array(mesh.points)  # Shape: (num_points, 2)
    vertices = np.array(
        [points[triangle] for triangle in triangles]
    )  # Shape: (n_triangles, 3, 2)
    return vertices.transpose(1, 0, 2)
