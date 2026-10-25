from pzr import learning
from pathlib import Path
from pzr.rtlola.actions import MPC_ACTION_NAMES

def export_geometry(output: Path) -> None:
    """Replay the starter traces with the teacher's actions and save the zonotopes."""
    task = learning.load_task("robot_arm", bundle="starter", budget=40)

    traces = task.validation_traces.traces + task.training_traces.traces
    samples = task.sample_metadata

    geometry = []

    for trace in traces:
        trace_id = f"{trace.trace_kind}::seed-{trace.seed}"
        trace_samples = samples[samples.trace_id == trace_id]
        teacher = dict(zip(trace_samples.step, trace_samples.teacher_action))

        session = learning.MonitorSession(trace, budget=task.budget)

        # Stop before the last event like the teacher so that the last event has no label
        while session.index < len(trace.events) - 1:
            obs = session.observe(geometry=True)
            step = session.index

            has_label = session.teacher_action is not None

            if obs.reduction_needed and not has_label:
                raise RuntimeError(f"{trace_id}, step {step}: reduction needed but teacher has no label")
            if has_label and not obs.reduction_needed:
                raise RuntimeError(f"{trace_id}, step {step}: teacher label but no reduction needed")

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

            if teacher[step] is not None and record["executed_action"] != teacher[step]:
                raise RuntimeError(
                    f"{trace_id}, step {step}: teacher chose {teacher[step]} but "
                    f"monitor ran {record['executed_action']} ({record['override_reason']})"
                )


if __name__ == "__main__":
    export_geometry(Path("geometry.json"))
