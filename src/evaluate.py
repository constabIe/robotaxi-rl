"""Numbers reported in report/report.typ.

uv run python src/evaluate.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from agents.autopilot import autopilot
from env.robotaxi import RobotaxiEnv
from env.vehicle import GRAVITY, Control, VehicleParams, VehicleState, simulate

KEY = jax.random.PRNGKey(0)
MAX_STEPS = 900


def braking_table() -> None:
    print("braking from 10 m/s: mu | simulated [m] | v0^2 / (2 min(a_max, mu g)) [m]")
    z = jnp.asarray(0.0)
    for mu in (1.0, 0.5, 0.1):
        p = VehicleParams(mu=mu)
        s = simulate(
            VehicleState(z, z, z, jnp.asarray(10.0), z, z, z),
            Control(jnp.asarray(-p.max_brake), z),
            15.0,
            3600,
            p,
        )
        print(
            f"  {mu:.1f} | {float(s.x):6.2f} | {100 / (2 * min(p.max_brake, mu * GRAVITY)):6.2f}"
        )


def episode(params, init_state, policy):
    def body(carry, key):
        s, ret, alive, length = carry
        _, s2, r, d, info = RobotaxiEnv.step(key, s, policy(s, key), params)
        out = (info["goal"] & alive, info["collision"] & alive)
        return (s2, ret + alive * r, alive & ~d, length + alive), out

    keys = jax.random.split(KEY, MAX_STEPS)
    init = (init_state, jnp.asarray(0.0), jnp.asarray(True), jnp.asarray(0))
    (_, ret, _, length), (goal, crash) = jax.lax.scan(body, init, keys)
    return ret, length, goal.any(), crash.any()


def random_policy(state, key):
    return jax.random.uniform(key, (2,), minval=-1, maxval=1)


def autopilot_policy(params):
    drive = partial(autopilot, params=params)
    return lambda state, key: drive(state)


def policy_table() -> None:
    print("episodes (900 steps max): map | mu | policy | return | steps | goal | crash")
    row = "  {} | {:.1f} | {:9s} | {:7.1f} | {:4d} | {} | {}"
    for map_id in (0, 1, 2):
        for mu in (1.0, 0.5, 0.1):
            params, s0 = RobotaxiEnv.init_params(
                KEY,
                map_id=map_id,
                max_steps=MAX_STEPS,
                path_length=100,
                vehicle=VehicleParams(mu=mu),
            )
            for name, policy in (
                ("autopilot", autopilot_policy(params)),
                ("random", random_policy),
            ):
                ret, n, goal, crash = episode(params, s0, policy)
                print(
                    row.format(
                        map_id, mu, name, float(ret), int(n), bool(goal), bool(crash)
                    )
                )


def throughput(n_envs: int = 1024, steps: int = 200) -> None:
    params, s0 = RobotaxiEnv.init_params(
        KEY, map_id=1, max_steps=MAX_STEPS, path_length=100
    )
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_envs, *jnp.shape(x))), s0)
    step = jax.vmap(RobotaxiEnv.step, in_axes=(None, 0, 0, None))

    @jax.jit
    def run(states, key):
        def body(s, k):
            a = jax.random.uniform(k, (n_envs, 2), minval=-1, maxval=1)
            return step(k, s, a, params)[1], None

        return jax.lax.scan(body, states, jax.random.split(key, steps))[0]

    jax.block_until_ready(run(states, KEY))
    t0 = time.perf_counter()
    jax.block_until_ready(run(states, jax.random.PRNGKey(1)))
    dt = time.perf_counter() - t0
    print(
        f"throughput ({jax.default_backend()}): {n_envs} envs x {steps} steps in {dt:.2f} s = {n_envs * steps / dt:,.0f} steps/s"
    )


if __name__ == "__main__":
    braking_table()
    policy_table()
    throughput()
