"""Run the robotaxi environment with keyboard control or the scripted autopilot.

Examples:
    uv run python src/main_robotaxi.py                       # drive with the arrow keys
    uv run python src/main_robotaxi.py --autopilot --mu 0.3  # scripted driver on a slippery road
    uv run python src/main_robotaxi.py --autopilot --record results/dry.mp4 --headless
"""

import argparse
import os
from functools import partial
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", type=int, default=1, help="map id from env/tasks.py")
    parser.add_argument(
        "--mu", type=float, default=1.0, help="tyre-road friction coefficient"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=900,
        help="episode length in agent steps (15 Hz)",
    )
    parser.add_argument(
        "--autopilot", action="store_true", help="use the scripted driver"
    )
    parser.add_argument("--record", type=Path, help="write an mp4 of the run")
    parser.add_argument("--episodes", type=int, help="stop after this many episodes")
    parser.add_argument(
        "--headless", action="store_true", help="no window, run as fast as possible"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import jax

    from agents.autopilot import autopilot
    from env.robotaxi import RobotaxiEnv
    from env.vehicle import VehicleParams
    from utils.autoreset import AutoResetWrapper
    from utils.robotaxi_renderer import RobotaxiFrontend

    params, init_state = RobotaxiEnv.init_params(
        jax.random.PRNGKey(0),
        map_id=args.map,
        max_steps=args.max_steps,
        path_length=100,
        perception_radius=5.0,
        num_ray_sensors=16,
        vehicle=VehicleParams(mu=args.mu),
    )
    env = AutoResetWrapper(RobotaxiEnv, params, init_state)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
    episodes = (
        args.episodes
        if args.episodes is not None
        else (1 if args.record and args.headless else None)
    )

    RobotaxiFrontend(
        env,
        params,
        init_state,
        eval_mode=args.autopilot,
        agent_fn=partial(autopilot, params=params),
        record_path=args.record,
        max_episodes=episodes,
        realtime=not args.headless,
    ).run()


if __name__ == "__main__":
    main()
