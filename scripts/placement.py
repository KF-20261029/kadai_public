"""Measure insertion using the actual holder rods and the colliding pen capsules."""

from dataclasses import dataclass

import mujoco
import numpy as np


CONTACT_TOLERANCE_M = 0.0005


@dataclass
class HolderGeometry:
    floor_z: float
    rim_z: float
    wall_centers: np.ndarray
    wall_radii: np.ndarray
    wall_bottom_z: np.ndarray
    wall_top_z: np.ndarray


def holder_geometry(sim):
    model = sim.model
    bottom = model.geom("holder_bottom").id
    wall_ids = [index for index in range(model.ngeom)
                if model.geom_bodyid[index] == sim.holder_id
                and (model.geom(index).name or "").startswith("holder_wall_")]
    if len(wall_ids) < 3:
        raise ValueError("Placement requires a holder with a bottom and at least three wall rods.")
    centers = model.geom_pos[wall_ids].copy()
    order = np.argsort(np.arctan2(centers[:, 1], centers[:, 0]))
    wall_ids = np.array(wall_ids)[order]
    centers = centers[order]
    for index in wall_ids:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[index])
        if not np.isclose(abs(rotation.reshape(3, 3)[2, 2]), 1):
            raise ValueError("This insertion metric requires vertical holder rods.")
    half_lengths = model.geom_size[wall_ids, 1]
    return HolderGeometry(
        floor_z=float(model.geom_pos[bottom, 2] + model.geom_size[bottom, 1]),
        rim_z=float(np.max(centers[:, 2] + half_lengths)),
        wall_centers=centers,
        wall_radii=model.geom_size[wall_ids, 0].copy(),
        wall_bottom_z=centers[:, 2] - half_lengths,
        wall_top_z=centers[:, 2] + half_lengths,
    )


def inside_wall_polygon(points, geometry, tolerance):
    vertices = geometry.wall_centers[:, :2]
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = points[:, None, :2] - vertices[None, :, :]
    distance = (edges[None, :, 0] * relative[:, :, 1]
                - edges[None, :, 1] * relative[:, :, 0]) / np.linalg.norm(edges, axis=1)[None, :]
    return (distance >= -tolerance).all(axis=1)


def classify_insertion(endpoint_a, endpoint_b, radii, geometry, tolerance=CONTACT_TOLERANCE_M):
    """Accept inserted pens, including those leaning against the holder wall.

    Both the lower end and the shaft at the opening must be inside the rod
    polygon. The lower cap must clear the actual rods and bottom, allowing
    0.5 mm for compliant contact penetration. At least one pen diameter must
    extend below the opening. Release and dwell are checked by the caller.
    """
    a, b = np.asarray(endpoint_a), np.asarray(endpoint_b)
    radii = np.asarray(radii)
    choose_a = a[:, 2] <= b[:, 2]
    lower = np.where(choose_a[:, None], a, b)
    upper = np.where(choose_a[:, None], b, a)
    height = upper[:, 2] - lower[:, 2]
    fraction = np.clip((geometry.rim_z - lower[:, 2]) / np.maximum(height, 1e-12), 0, 1)
    opening_point = lower + fraction[:, None] * (upper - lower)
    nearest_wall_z = np.clip(lower[:, 2, None], geometry.wall_bottom_z, geometry.wall_top_z)
    distance = np.sqrt(
        ((lower[:, None, :2] - geometry.wall_centers[None, :, :2]) ** 2).sum(axis=2)
        + (lower[:, 2, None] - nearest_wall_z) ** 2
    )
    wall_clearance = (distance - geometry.wall_radii[None, :] - radii[:, None]).min(axis=1)
    bottom_clearance = lower[:, 2] - radii - geometry.floor_z
    insertion_depth = geometry.rim_z - (lower[:, 2] - radii)
    lower_inside = inside_wall_polygon(lower, geometry, tolerance)
    opening_inside = inside_wall_polygon(opening_point, geometry, tolerance)
    result = (lower_inside & opening_inside & (wall_clearance >= -tolerance)
              & (bottom_clearance >= -tolerance) & (insertion_depth >= 2 * radii))
    details = {
        "lower_endpoint_m": lower,
        "opening_point_m": opening_point,
        "wall_clearance_m": wall_clearance,
        "bottom_clearance_m": bottom_clearance,
        "insertion_depth_m": insertion_depth,
        "lower_endpoint_inside": lower_inside,
        "opening_inside": opening_inside,
    }
    return result, details


def pen_placement(sim, return_details=False):
    if not hasattr(sim, "_placement_geometry"):
        sim._placement_geometry = holder_geometry(sim)
    ids = [sim.model.geom(sim.model.body(int(body)).name + "_barrel").id for body in sim.pen_ids]
    centers = sim.data.geom_xpos[ids]
    axes = sim.data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
    half_lengths = sim.model.geom_size[ids, 1]
    radii = sim.model.geom_size[ids, 0]
    holder_position = sim.data.xpos[sim.holder_id]
    holder_rotation = sim.data.xmat[sim.holder_id].reshape(3, 3)
    endpoint_a = (centers - axes * half_lengths[:, None] - holder_position) @ holder_rotation
    endpoint_b = (centers + axes * half_lengths[:, None] - holder_position) @ holder_rotation
    result, details = classify_insertion(endpoint_a, endpoint_b, radii, sim._placement_geometry)
    return (result, details) if return_details else result
