"""Reward function of the robotaxi task.

    r_t = w_p · clip(Φ(s_{t+1}) − Φ(s_t), ±Δ_max)      progress (potential-based, Φ = −D)
          − w_lat · e_lat²                            stay on the planned path
          − w_v · max(0, v_x − v_lim)²                 speed limit
          − w_u · ‖u_t − u_{t−1}‖²                     smooth control
          − w_ay · max(0, |a_y| − a_y,comf)²           passenger comfort
          − w_t                                        time cost
          + R_goal · 1[goal] − R_crash (1 + v_x/v_lim) · 1[collision ∨ off-map]

D is the remaining path length in metres; all terms are per environment step.
"""

from typing import NamedTuple

import equinox
import jax
from jax import numpy as jnp


class RewardWeights(equinox.Module):
    progress: float = 1.0  # w_p [1/m]
    max_progress: float = (
        2.0  # Δ_max [m per step], guards against path re-planning jumps
    )
    lateral: float = 0.002  # w_lat [1/m^2]
    speed_limit: float = 10.0  # v_lim [m/s] (36 km/h)
    overspeed: float = 0.01  # w_v [s^2/m^2]
    smooth: float = 0.02  # w_u
    comfort_accel: float = 3.0  # a_y,comf [m/s^2]
    comfort: float = 0.01  # w_ay [s^4/m^2]
    time: float = 0.01  # w_t
    goal: float = 10.0  # R_goal
    crash: float = 5.0  # R_crash


class RewardInputs(NamedTuple):
    remaining_before: jax.Array  # D(s_t) on the current path [m]
    remaining_after: jax.Array  # D(s_{t+1}) on the same path [m]
    lateral: jax.Array  # e_lat [m]
    v_x: jax.Array  # [m/s]
    lat_accel: jax.Array  # a_y [m/s^2]
    action: jax.Array  # u_t in [-1, 1]^2
    prev_action: jax.Array  # u_{t-1}
    goal: jax.Array  # bool
    crash: jax.Array  # bool (collision or left the map)


def reward_terms(x: RewardInputs, w: RewardWeights) -> dict[str, jax.Array]:
    """Individual reward components (already signed and weighted)."""
    progress = jnp.clip(
        x.remaining_before - x.remaining_after, -w.max_progress, w.max_progress
    )
    return {
        "r_progress": w.progress * progress,
        "r_lateral": -w.lateral * x.lateral**2,
        "r_overspeed": -w.overspeed * jnp.maximum(0.0, x.v_x - w.speed_limit) ** 2,
        "r_smooth": -w.smooth * jnp.sum((x.action - x.prev_action) ** 2),
        "r_comfort": -w.comfort
        * jnp.maximum(0.0, jnp.abs(x.lat_accel) - w.comfort_accel) ** 2,
        "r_time": -w.time * jnp.ones_like(x.v_x),
        "r_goal": w.goal * x.goal.astype(jnp.float32),
        "r_crash": -w.crash
        * (1.0 + x.v_x / w.speed_limit)
        * x.crash.astype(jnp.float32),
    }


def total_reward(terms: dict[str, jax.Array]) -> jax.Array:
    return sum(terms.values())
