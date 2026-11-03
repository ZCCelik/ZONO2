"""Replay the starter traces and save the zonotope geometry retrieved by teacher action"""
import numpy as np
import pandas as pd

from pzr import learning
from pathlib import Path
from dataclasses import replace

from pzr.learning.geometry import GeometrySample, GeometrySampleDataset, save_geometry
from pzr.learning.tasks import Task
from pzr.rtlola.actions import MPC_ACTION_NAMES
from pzr.rtlola.scenarios import RtlolaTrace


def export_geometry(output: Path, limit: int | None = None, events: int | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    task = learning.load_task("robot_arm", bundle="starter", budget=40)
    # For easy debugging. Limit the trace and event number
    traces = task.training_traces.traces + task.validation_traces.traces
    traces = traces[:limit]
    traces = [replace(t, events=t.events[:events]) for t in traces]
    samples = task.sample_metadata
    geometry = []
    for trace in traces:
        geometry.extend(replay_trace(trace, samples, task))
    is_complete = limit is None and events is None
    dataset = GeometrySampleDataset(tuple(geometry))
    _check_feature_alignment(dataset, task, complete=is_complete)
    save_geometry(dataset, output, metadata={
        "task": "robot_arm",
        "bundle": "starter",
        "budget": task.budget,
        "limit": limit,
        "events": events,
        "complete": limit is None and events is None})


def replay_trace(trace: RtlolaTrace,
                 samples: pd.DataFrame,
                 task: Task) -> list[GeometrySample] :
    trace_id = f"{trace.trace_kind}:seed-{trace.seed}"
    trace_samples = samples[samples.trace_id == trace_id]
    teacher = dict(zip(trace_samples.step, trace_samples.teacher_action))
    sample_id_by_step = dict(zip(trace_samples.step, trace_samples.sample_id))
    geometry = []
    session = learning.MonitorSession(trace, budget=task.budget)
    # Stop before the last event like the teacher so that the last event has no label
    while session.index < len(trace.events) - 1:
        obs = session.observe(geometry=True)
        step = session.index
        has_label = step in teacher
        if obs.reduction_needed and not has_label:
            raise RuntimeError(f"{trace_id}, step {step}: reduction needed but teacher has no label")
        if has_label and not obs.reduction_needed:
            raise RuntimeError(f"{trace_id}, step {step}: teacher label but no reduction needed")
        if obs.reduction_needed and step in teacher:
            geometry.append(
                GeometrySample(trace_id,
                               sample_id_by_step[step],
                               step,
                               teacher[step],
                               obs.features,
                               obs.dynamic, obs.total)
            )
            action = MPC_ACTION_NAMES.index(teacher[step])
        else:
            action = 0  # ignored under budget
        record = session.step(action)
        if teacher.get(step) is not None and record["executed_action"] != teacher[step]:
            raise RuntimeError(
                f"{trace_id}, step {step}: teacher chose {teacher[step]} but "
                f"monitor ran {record['executed_action']} ({record['override_reason']})"
            )
    return geometry


def _check_feature_alignment(dataset: GeometrySampleDataset, task: Task, complete: bool) -> None:
    """Replayed Geometry15 features must exactly reproduce the teacher's recorded features."""
    row_by_id = {sid: i for i, sid in enumerate(task.dataset.sample_ids)}
    missing = [g.sample_id for g in dataset.samples if g.sample_id not in row_by_id]
    if missing:
        raise ValueError(f"{len(missing)} samples not in the teacher dataset, e.g. {missing[0]}")
    if complete and set(row_by_id) != {g.sample_id for g in dataset.samples}:
        raise ValueError("complete export does not cover every teacher sample")
    expected = task.dataset.features[[row_by_id[g.sample_id] for g in dataset.samples]]
    replayed = dataset.features
    if replayed.shape != expected.shape:
        raise ValueError(f"feature shape {replayed.shape} differs from teacher {expected.shape}")
    if not np.array_equal(replayed, expected, equal_nan=True):
        bad = np.flatnonzero((replayed != expected).any(axis=1))
        raise ValueError(f"replayed features differ from teacher for {len(bad)} samples, "
                         f"e.g. {dataset.samples[bad[0]].sample_id}")