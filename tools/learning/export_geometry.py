

import numpy as np
import pandas as pd

from pzr import learning
from pathlib import Path
from dataclasses import replace, dataclass

from pzr.learning.monitor import Zonotope
from pzr.learning.tasks import Task
from pzr.rtlola.actions import MPC_ACTION_NAMES
from pzr.rtlola.scenarios import RtlolaTrace

@dataclass(frozen=True)
class Geometry:
    trace_id: str
    step: int
    teacher_action: str
    features: np.ndarray
    dynamic: Zonotope | None = None
    total: Zonotope | None = None

def export_geometry(output: Path,  limit: int | None = None, events: int | None = None) -> None:
    """Replay the starter traces with the teacher's actions
    and save the zonotopes in a list of Geometry objects."""

    task = learning.load_task("robot_arm", bundle="starter", budget=40)

    # For easy debugging. Limit the trace and event number
    traces = task.validation_traces.traces + task.training_traces.traces
    traces = traces[:limit]
    traces = [replace(t, events=t.events[:events]) for t in traces]

    samples = task.sample_metadata
    geometry = []

    for trace in traces:
        geometry.extend(replay_trace(trace, samples, task))

    print("collected", len(geometry), "samples from", len(traces), "traces")


def replay_trace(trace: RtlolaTrace,
                 samples: pd.DataFrame,
                 task: Task) -> list[Geometry] :

    trace_id = f"{trace.trace_kind}:seed-{trace.seed}"
    trace_samples = samples[samples.trace_id == trace_id]
    teacher = dict(zip(trace_samples.step, trace_samples.teacher_action))
    geometry = []

    print(trace_id, "samples:", len(trace_samples), "labels:", len(teacher))

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

        print(step, ", reduction needed:", obs.reduction_needed,
              "gens:", obs.dynamic.generators.shape,
              "teacher:", teacher.get(step))

        if obs.reduction_needed and step in teacher:
            geometry.append(
                Geometry(trace_id, step,
                teacher[step],
                obs.features,
                obs.dynamic, obs.total
                )
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("geometry.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--events", type=int, default=None)
    args = parser.parse_args()
    # debug config --limit 1 --events 10
    export_geometry(args.output, limit=args.limit, events=args.events)