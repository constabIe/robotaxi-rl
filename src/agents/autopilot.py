"""Scripted driver: pure-pursuit steering on the planned path + speed P-controller.

Used to record reproducible videos and as a sanity baseline for the reward
(it should collect a clearly higher return than a random policy).
"""

from functools import partial

import jax
from jax import numpy as jnp

from env.robotaxi import RobotaxiParams, RobotaxiState, cast_rays
from utils.geometry import path_frame, point_at_arc, to_body


@partial(jax.jit, static_argnames=("params", "cruise_speed"))
def autopilot(
    state: RobotaxiState,
    params: RobotaxiParams,
    cruise_speed: float = 5.0,
) -> jax.Array:
    """Return a normalised action (a, δ) ∈ [-1, 1]²."""
    vp, scale = params.vehicle, params.meters_per_cell
    pos, path = state.agent_pos, state.path_array

    # Pure pursuit: δ = atan(2 L sin α / l_d) towards the path point l_d metres ahead
    # (along the path). A short, speed-proportional l_d limits corner cutting.
    frame = path_frame(pos, path)
    l_ahead = jnp.clip(0.8 * state.v_x, 3.0, 8.0)
    target = point_at_arc(path, frame.arc + l_ahead / scale)
    d = to_body(target - pos, state.yaw) * scale
    l_d = jnp.maximum(jnp.linalg.norm(d), 1.0)
    alpha = jnp.arctan2(d[1], d[0])
    steer = jnp.arctan(2.0 * vp.wheelbase * jnp.sin(alpha) / l_d)

    # Speed reference: lateral-acceleration limit on the pursuit arc and braking distance to obstacles ahead.
    turn_radius = l_d / (2.0 * jnp.maximum(jnp.abs(jnp.sin(alpha)), 1e-3))
    v_curve = jnp.sqrt(2.0 * turn_radius)  # a_y ≤ 2 m/s²
    obstacles = jnp.concatenate(
        [state.static_obstacles, state.kinematic_obstacles], axis=0
    )
    rays, dists = cast_rays(pos, state.agent_forward_dir, obstacles, params)
    # Ray distances are already measured to the obstacle surface (collider circle).
    forward = jnp.sum(rays * state.agent_forward_dir, axis=-1) > jnp.cos(0.25)
    free = (jnp.min(jnp.where(forward, dists, params.perception_radius)) - 0.2) * scale
    v_obst = jnp.sqrt(2.0 * 0.5 * vp.max_brake * jnp.maximum(free, 0.0))
    v_ref = jnp.minimum(cruise_speed, jnp.minimum(v_curve, v_obst))

    accel = jnp.clip(1.5 * (v_ref - state.v_x), -vp.max_brake, vp.max_accel)
    a_norm = jnp.where(accel >= 0.0, accel / vp.max_accel, accel / vp.max_brake)
    return jnp.stack([a_norm, steer / vp.max_steer]).clip(-1.0, 1.0)
