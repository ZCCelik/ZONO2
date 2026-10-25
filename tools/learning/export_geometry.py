from pzr import learning
from pathlib import Path
from dataclasses import replace
from pzr.rtlola.actions import MPC_ACTION_NAMES

def export_geometry(output: Path,  limit: int | None = None, events: int | None = None) -> None:
    """Replay the starter traces with the teacher's actions and save the zonotopes."""
    task = learning.load_task("robot_arm", bundle="starter", budget=40)

    traces = task.validation_traces.traces + task.training_traces.traces
    traces = traces[:limit]
    traces = [replace(t, events=t.events[:events]) for t in traces]

    samples = task.sample_metadata

    geometry = []

    for trace in traces:
        trace_id = f"{trace.trace_kind}:seed-{trace.seed}"
        trace_samples = samples[samples.trace_id == trace_id]
        teacher = dict(zip(trace_samples.step, trace_samples.teacher_action))

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

                geometry.append({
                    "trace_id": trace_id, "step": step,
                    "teacher_action": teacher[step],
                    "dynamic": obs.dynamic, "total": obs.total,
                    "features": obs.features,
                })
                action = MPC_ACTION_NAMES.index(teacher[step])
            else:
                action = 0  # ignored under budget

            record = session.step(action)

            if teacher.get(step) is not None and record["executed_action"] != teacher[step]:
                raise RuntimeError(
                    f"{trace_id}, step {step}: teacher chose {teacher[step]} but "
                    f"monitor ran {record['executed_action']} ({record['override_reason']})"
                )

    print("collected", len(geometry), "samples from", len(traces), "traces")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("geometry.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--events", type=int, default=None)
    args = parser.parse_args()
    # debug config --limit 1 --events 10
    export_geometry(args.output, limit=args.limit, events=args.events)