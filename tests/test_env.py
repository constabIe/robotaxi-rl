"""RobotaxiEnv: interface, JAX transformations, numerical safety, reward sanity."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from agents.autopilot import autopilot
from env.robotaxi import RobotaxiEnv, RobotaxiObservation, RobotaxiState
from utils.autoreset import AutoResetWrapper

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def env():
    return RobotaxiEnv.init_params(KEY, map_id=1, max_steps=900, path_length=100)


def rollout(params, state, policy, steps):
    """Episode return and final info under ``policy(state, key)``; stops at the first done."""

    def body(carry, key):
        s, ret, alive = carry
        _, s2, r, d, info = RobotaxiEnv.step(key, s, policy(s, key), params)
        return (s2, ret + alive * r, alive & ~d), info

    keys = jax.random.split(KEY, steps)
    (_, ret, _), infos = jax.lax.scan(
        body, (state, jnp.asarray(0.0), jnp.asarray(True)), keys
    )
    return ret, infos


def test_interface(env):
    params, init_state = env
    obs, state = RobotaxiEnv.reset(KEY, params, init_state)
    assert isinstance(obs, RobotaxiObservation) and isinstance(state, RobotaxiState)
    obs, state, reward, done, info = RobotaxiEnv.step(
        KEY, state, jnp.array([0.5, 0.0]), params
    )
    assert obs.as_vector().shape == (39,)
    assert reward.shape == () and done.dtype == jnp.bool_
    assert {"terminated", "truncated", "r_progress", "time"} <= info.keys()


def test_autoreset_wrapper(env):
    params, init_state = env
    wrapped = AutoResetWrapper(RobotaxiEnv, params, init_state)
    obs, state = wrapped.reset(KEY)
    for _ in range(3):
        obs, state, *_ = wrapped.step(KEY, state, jnp.array([1.0, 0.0]))
    assert bool(jnp.all(jnp.isfinite(obs.as_vector())))


def test_vmap_scan_no_nan(env):
    params, init_state = env
    n, steps = 128, 200
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), init_state)
    actions = jax.random.uniform(KEY, (steps, n, 2), minval=-1.0, maxval=1.0)
    step = jax.vmap(RobotaxiEnv.step, in_axes=(None, 0, 0, None))

    def body(s, a):
        obs, s, r, _, _ = step(KEY, s, a, params)
        return s, (obs.as_vector(), r)

    _, (obs, rewards) = jax.jit(lambda s: jax.lax.scan(body, s, actions))(states)
    assert obs.shape == (steps, n, 39)
    assert bool(jnp.all(jnp.isfinite(obs))) and bool(jnp.all(jnp.isfinite(rewards)))


def test_deterministic(env):
    params, init_state = env
    a = jnp.array([0.7, 0.2])
    s1 = RobotaxiEnv.step(KEY, init_state, a, params)[1]
    s2 = RobotaxiEnv.step(KEY, init_state, a, params)[1]
    assert jax.tree.all(jax.tree.map(lambda x, y: bool(jnp.array_equal(x, y)), s1, s2))


def test_standing_still_is_not_rewarded(env):
    params, init_state = env
    ret, infos = rollout(params, init_state, lambda s, k: jnp.array([-1.0, 0.0]), 900)
    assert float(ret) < 0.0
    assert bool(infos["truncated"][-1]) and not bool(infos["terminated"].any())


def test_autopilot_beats_random(env):
    params, init_state = env
    drive = partial(autopilot, params=params)
    ret_auto, infos = rollout(params, init_state, lambda s, k: drive(s), 900)
    ret_rand, _ = rollout(
        params,
        init_state,
        lambda s, k: jax.random.uniform(k, (2,), minval=-1, maxval=1),
        900,
    )
    assert bool(infos["goal"].any())
    assert float(ret_auto) > float(ret_rand) + 50.0
