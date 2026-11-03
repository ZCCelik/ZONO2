import numpy as np
import pandas as pd
import json

from dataclasses import dataclass
from typing import Mapping
from pathlib import Path

from pzr.learning.monitor import Zonotope
from pzr.artifact_io import write_csv_atomic, write_json_atomic

GEOMETRY_SCHEMA = "pzr.starter-geometry.v1"


@dataclass(frozen=True)
class GeometrySample:
    trace_id: str
    sample_id: str
    step: int
    teacher_action: str
    features: np.ndarray
    dynamic: Zonotope
    total: Zonotope


@dataclass(frozen=True)
class GeometrySampleDataset:
    """Pre-event zonotopes at teacher decisions, aligned with the starter samples."""

    samples: tuple[GeometrySample, ...]

    def __post_init__(self) -> None:
        ids = [g.sample_id for g in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("sample identifiers must be unique")
        for g in self.samples:
            if g.dynamic.center.shape != g.total.center.shape:
                raise ValueError(f"{g.sample_id}: dynamic and total dimensions differ")

    def __len__(self):
        return len(self.samples)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def features(self) -> np.ndarray:
        return np.stack([g.features for g in self.samples])   # (N, 15) when you need the block


def save_geometry(dataset:GeometrySampleDataset, output: Path, metadata: Mapping[str, object]) -> None:
    """Write geometry.npz, samples.csv and manifest.json for the exported zonotopes.

    Centers are stacked to (N, d). Generator matrices have different column counts,
    so they are concatenated to (d, total columns); sample i's generators are
    generators[:, offsets[i]:offsets[i + 1]].
    """
    temporary = output / ".geometry.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,
                            sample_ids=np.asarray([s.sample_id for s in dataset.samples]),
                            trace_ids=np.asarray([s.trace_id for s in dataset.samples]),
                            features=dataset.features,
                            steps=np.asarray([s.step for s in dataset.samples], dtype=np.int64),
                            teacher_actions=np.asarray([s.teacher_action for s in dataset.samples]),
                            dynamic_centers=np.stack([s.dynamic.center for s in dataset.samples]),
                            dynamic_generators=np.concatenate([s.dynamic.generators for s in dataset.samples], axis=1),
                            dynamic_offsets=np.cumsum([0] + [s.dynamic.generators.shape[1] for s in dataset.samples]),
                            total_centers=np.stack([s.total.center for s in dataset.samples]),
                            total_generators=np.concatenate([s.total.generators for s in dataset.samples], axis=1),
                            total_offsets=np.cumsum([0] + [s.total.generators.shape[1] for s in dataset.samples]),
                            )
    temporary.replace(output / "geometry.npz")
    geometry_metadata = pd.DataFrame({
        "sample_id": [s.sample_id for s in dataset.samples],
        "trace_id": [s.trace_id for s in dataset.samples],
        "step": [s.step for s in dataset.samples],
        "teacher_action": [s.teacher_action for s in dataset.samples],
        "dynamic_rows": [s.dynamic.generators.shape[0] for s in dataset.samples],
        "dynamic_generators": [s.dynamic.generators.shape[1] for s in dataset.samples],
        "total_generators": [s.total.generators.shape[1] for s in dataset.samples]
    })
    write_csv_atomic(geometry_metadata, output / "samples.csv")
    write_json_atomic({
        "schema": GEOMETRY_SCHEMA,
        "num_samples": len(dataset),
        **dict(metadata),
    }, output / "manifest.json")


def load_geometry(directory: Path) -> tuple[GeometrySampleDataset, pd.DataFrame, dict[str, object]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("unsupported geometry dataset schema")
    with np.load(directory / "geometry.npz", allow_pickle=False) as arrays:
        d_off, t_off = arrays["dynamic_offsets"], arrays["total_offsets"]
        d_gen, t_gen = arrays["dynamic_generators"], arrays["total_generators"]
        samples = []
        for i in range(len(arrays["sample_ids"])):
            samples.append(GeometrySample(
                trace_id=str(arrays["trace_ids"][i]),
                sample_id=str(arrays["sample_ids"][i]),
                step=int(arrays["steps"][i]),
                teacher_action=str(arrays["teacher_actions"][i]),
                features=arrays["features"][i],
                dynamic=Zonotope(arrays["dynamic_centers"][i], d_gen[:, d_off[i]:d_off[i + 1]]),
                total=Zonotope(arrays["total_centers"][i], t_gen[:, t_off[i]:t_off[i + 1]]),
            ))
        dataset = GeometrySampleDataset(tuple(samples))
    if len(dataset) != int(manifest["num_samples"]):
        raise ValueError("geometry manifest sample count differs")
    sample_metadata = pd.read_csv(directory / "samples.csv")
    _validate_sample_metadata(dataset, sample_metadata)
    return dataset, sample_metadata, manifest


def _validate_sample_metadata(dataset: GeometrySampleDataset, metadata: pd.DataFrame) -> None:
    required = {"sample_id", "trace_id", "step", "teacher_action", "dynamic_rows", "dynamic_generators", "total_generators"}
    if not required <= set(metadata.columns):
        raise ValueError(f"sample metadata lacks columns: {sorted(required - set(metadata))}")
    if len(metadata) != dataset.num_samples:
        raise ValueError("sample metadata row count differs from dataset")
    if tuple(metadata["sample_id"].astype(str)) != tuple(gs.sample_id for gs in dataset.samples):
        raise ValueError("sample metadata identifiers are not aligned")
    metadata_teacher_action = tuple(zip(metadata["sample_id"].astype(str),
                                        metadata["step"].astype(int),
                                        metadata["teacher_action"].astype(str)))
    dataset_teacher_action = tuple((d.sample_id, d.step, d.teacher_action) for d in dataset.samples)
    if metadata_teacher_action != dataset_teacher_action:
        raise ValueError("sample metadata steps or teacher actions are not aligned")
