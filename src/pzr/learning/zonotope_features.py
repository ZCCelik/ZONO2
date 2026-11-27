from dataclasses import dataclass

import numpy as np

from pzr.learning.monitor import Zonotope
from numpy.typing import NDArray

GENERATOR_FEATURE_NAMES = (
    "l1_norm",
    "l2_norm",
    "inf_norm",
    "girard_score",            # l1_norm - inf_norm
    "off_axis_fraction",       # girard_score / l1_norm
    "relative_l2_norm",        # l2_norm / mean active l2_norm
    "row_support_fraction",    # share of rows with a nonzero entry
    "is_axis_aligned",         # exactly one nonzero entry
    "is_active",               # l2_norm > atol
    "max_abs_cosine",          # highest |cos| with any other generator
)
COORDINATE_FEATURE_NAMES = (
    "center",
    "dynamic_radius",          # sum of |G[i, :]| over dynamic generators
    "total_radius",            # same over the total zonotope
    "slack_fraction",          # (total_radius - dynamic_radius) / total_radius
    "width_share",             # dynamic_radius / sum of all dynamic radii
    "generator_support_fraction",  # share of generators touching this row
    "max_coefficient_share",   # largest |G[i, j]| / dynamic_radius
    "is_active",               # row has any nonzero entry
)
GENERATOR_GROUPS = {
    "size":      ("l1_norm", "l2_norm", "inf_norm", "relative_l2_norm"),
    "shape":     ("girard_score", "off_axis_fraction", "is_axis_aligned", "row_support_fraction"),
    "coupling":  ("max_abs_cosine",),
}
COORDINATE_GROUPS = {
    "width":     ("dynamic_radius", "total_radius", "width_share", "max_coefficient_share"),
    "slack":     ("slack_fraction",),
}


@dataclass(frozen=True)
class ZonotopeGraph:

    coord_features: NDArray[np.float32]
    gen_features: NDArray[np.float32]
    edge_index: NDArray[np.int32]
    edge_weight: NDArray[np.float32]


def get_generator_features(zonotope: Zonotope, atol: float = 1e-12) -> NDArray[np.float32]:
    """One row per generator (column of G); columns follow GENERATOR_FEATURE_NAMES."""
    generators = zonotope.generators  # shape (d, m), columns are generators
    num_rows, num_generators = generators.shape
    if num_rows == 0 or num_generators == 0:
        return np.zeros((num_generators, len(GENERATOR_FEATURE_NAMES)), dtype=np.float32)
    absolute = np.abs(generators)
    l1_norm = absolute.sum(axis=0)
    l2_norm = np.linalg.norm(generators, axis=0)
    inf_norm = absolute.max(axis=0)
    girard_score = l1_norm - inf_norm
    nonzero_per_generator = np.count_nonzero(absolute > atol, axis=0)
    active = nonzero_per_generator > 0
    off_axis_fraction = np.zeros(num_generators)
    off_axis_fraction[active] = girard_score[active] / l1_norm[active]
    relative_l2_norm = np.zeros(num_generators)
    if active.any():
        relative_l2_norm[active] = l2_norm[active] / l2_norm[active].mean()
    row_support_fraction = nonzero_per_generator / num_rows
    is_axis_aligned = nonzero_per_generator == 1
    max_abs_cosine = np.zeros(num_generators)
    if np.count_nonzero(active) >= 2:
        unit = generators[:, active] / l2_norm[active]
        # for unit vectors, cos = dot product
        cos_abs = np.abs(unit.T @ unit)
        np.fill_diagonal(cos_abs, 0.0)
        # ignore cosine with itself
        max_abs_cosine[active] = cos_abs.max(axis=0)
    features = np.stack([
        l1_norm,
        l2_norm,
        inf_norm,
        girard_score,
        off_axis_fraction,
        relative_l2_norm,
        row_support_fraction,
        is_axis_aligned,
        active,
        max_abs_cosine,
    ], axis=1)
    return features.astype(np.float32)


def get_coordinate_features(dynamic: Zonotope, total: Zonotope, atol: float = 1e-12):
    d = dynamic.generators.shape[0]
    dynamic_radius = np.abs(dynamic.generators).sum(axis=1)
    total_radius = np.abs(total.generators).sum(axis=1)
    slack_fraction = np.zeros(dynamic_radius.shape)
    has_total = total_radius > atol
    slack_fraction[has_total] = (total_radius[has_total] - dynamic_radius[has_total]) / total_radius[has_total]
    width_share = np.zeros(d)
    sum_all_dynamic_radius = np.sum(dynamic_radius)
    if sum_all_dynamic_radius > atol:
        width_share = dynamic_radius / sum_all_dynamic_radius
    touching = np.abs(dynamic.generators) > atol
    nonzero_per_generator = touching.sum(axis=0)
    active_generators = nonzero_per_generator > 0
    generator_support_fraction = np.zeros(d)
    if active_generators.any():
        generator_support_fraction = touching.sum(axis=1) / active_generators.sum()
    active_rows = dynamic_radius > atol
    max_coefficient_share = np.zeros(d)
    if active_rows.any():
        max_coefficient_share[active_rows] = np.abs(dynamic.generators[active_rows]).max(axis=1) / dynamic_radius[active_rows]
    features = np.stack([
        dynamic.center,
        dynamic_radius,
        total_radius,
        slack_fraction,
        width_share,
        generator_support_fraction,
        max_coefficient_share,
        active_rows,
    ], axis=1)
    return features.astype(np.float32)
