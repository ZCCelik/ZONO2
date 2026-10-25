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

        while not session.done:
            obs = session.observe(geometry=True)
            step = session.index

            if obs.reduction_needed and step in teacher:
                geometry.append({
                    "trace_id": trace_id, "step": step,
                    "teacher_action": teacher[step],
                    "dynamic": obs.dynamic, "total": obs.total,
                    "features": obs.features,
                })
                action = MPC_ACTION_NAMES.index(teacher[step])
            else:
                action = 0

            record = session.step(action)

            if step in teacher and record["executed_action"] != teacher[step]:
                print("left the teacher's path at step", step, record)
                break


if __name__ == "__main__":
    export_geometry(Path("geometry.json"))
