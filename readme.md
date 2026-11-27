# Robotaxi RL environment

IAEML (Innopolis University), Assignment 1: a JAX robotaxi environment built on the course's `BaseEnv`.
It has single-track vehicle dynamics with tyre friction, a body-frame observation and a potential-based reward.

Report: [report/report.pdf](report/report.pdf) (source: [report/report.typ](report/report.typ)).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11 is installed automatically).

```bash
uv sync
```

On macOS add `JAX_PLATFORM_NAME=cpu` before the commands below if JAX complains.

## Usage

```bash
uv run python src/main_robotaxi.py                        # drive with the arrow keys
uv run python src/main_robotaxi.py --autopilot --mu 0.5   # scripted driver, wet road
uv run python src/main_robotaxi.py --autopilot --mu 1.0 --record results/dry.mp4 --headless
uv run python src/evaluate.py                             # numbers used in the report
uv run pytest                                             # tests
```

Options of `main_robotaxi.py`: `--map {0,1,2}`, `--mu` (friction), `--max-steps`, `--episodes`.
Videos are in [results/](results/).

## Layout

| Path | Contents |
|---|---|
| `src/env/robotaxi.py` | `RobotaxiEnv(BaseEnv)`: state, observation, step, termination |
| `src/env/vehicle.py` | vehicle dynamics |
| `src/env/rewards.py` | reward terms |
| `src/agents/autopilot.py` | scripted baseline driver |
| `src/utils/robotaxi_renderer.py` | pygame renderer and video recording |
| `src/env/base.py`, `src/env/tasks.py`, `src/utils/*`, `src/main.py` | course starter code ([upstream](https://github.com/InnoRL/IAEML-assignment)) |
