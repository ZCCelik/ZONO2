from dataclasses import dataclass

import numpy as np
from cmath import inf
from sympy.series import order

from pzr.learning.geometry import GeometrySample, GeometrySampleDataset
from pzr.learning.monitor import Zonotope
from pzr.rtlola.features import _mean_generator_off_axis_fraction
from numpy.typing import NDArray

from pzr.rtlola.robot_arm import generate_robot_arm_events

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

@dataclass(frozen=True)
class ZonotopeGraph:

    coord_features: NDArray[np.float32]
    gen_features: NDArray[np.float32]
    edge_index: NDArray[np.int32]
    edge_weight: NDArray[np.float32]

# TODO:: decide on whether use float32 or 64
def get_generator_features(zonotope: Zonotope, budget: int, atol: float = 1e-12) -> NDArray[np.float32]:
    generators = zonotope.generators
    l1_norm = (
        np.linalg.norm(generators, ord=1, axis=0)
        if generators.size else np.zeros(0, dtype=np.float64)
    )
    l2_norm = (
        np.linalg.norm(generators, axis=0)
        if generators.size else np.zeros(0, dtype=np.float64)
    )
    inf_norm = (
        np.linalg.norm(generators, ord=inf, axis=0)
        if generators.size else np.zeros(0, dtype=np.float64)
    )
    girard_score = l1_norm - inf_norm
    off_axis_fraction = girard_score / l1_norm
    active = l2_norm > atol
    active_generators = generators[:, active]
    relative_l2_norm = l2_norm / np.mean(active_generators, axis=0)
    num_rows, num_columns = generators.shape
    non_zero_count = np.count_nonzero(a=generators[generators>atol], axis=0, keepdims=True)
    row_support_fraction = non_zero_count / num_rows
    is_axis_aligned = non_zero_count[non_zero_count==1].astype(int)
    scalar_product = generators * generators.T
    normalised_generators = active_generators / l2_norm[active]
    # for normalized vectors the cos is simply the dot product
    cos_abs = np.abs(normalised_generators.T @ normalised_generators)
    # ignore each generator's cosine with itself
    np.fill_diagonal(cos_abs, 0)
    max_abs_cosine = np.zeros((1,num_columns))

    if active_generators.shape[1] > 2:
        # cos_abs is symmetric, you can choose between the axis
        max_abs_cosine[active] = cos_abs.max(axis=0)

    features = np.asarray([
        l1_norm,
        l2_norm,
        inf_norm,
        girard_score,
        off_axis_fraction,
        relative_l2_norm,
        row_support_fraction,
        is_axis_aligned,
        generators > atol,
        max_abs_cosine
    ], np.float32)

    return features
    """
        ###############-TEST-################
a1 = [1,2,3]
a2 = [10, 200, 3000]

    """