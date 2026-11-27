"""Planar vehicle dynamics: single-track (bicycle) model with saturating tyres.

All quantities are SI (metres, seconds, radians, newtons). Every function is
pure and jit/vmap friendly; parameters are hashable Python floats so that they
can be passed to ``jax.jit`` as static arguments.

Model summary (body frame, x forward, y left):

    v̇x = (Fx − Fyf sin δ)/m + vy r
    v̇y = (Fyr + Fyf cos δ)/m − vx r
    ṙ  = (lf Fyf cos δ − lr Fyr)/Iz

with tyre forces Fy = μ Fz tanh(cs α), longitudinal force
Fx = m a − Crr m g tanh(vx/v0) − ½ ρ CdA vx², |Fx| ≤ μ m g, and a smooth blend
with the kinematic model below ``v_blend_max`` to avoid the 1/vx singularity.
"""

from typing import NamedTuple

import equinox
import jax
from jax import numpy as jnp

GRAVITY = 9.81
AIR_DENSITY = 1.2


class VehicleParams(equinox.Module):
    """Physical parameters. Defaults: CommonRoad vehicle 2 (BMW 320i)."""

    mass: float = 1093.3  # m [kg]
    inertia_z: float = 1791.6  # Iz [kg m^2]
    l_f: float = 1.156  # CoG -> front axle [m]
    l_r: float = 1.423  # CoG -> rear axle [m]
    c_s: float = 20.89  # normalised cornering stiffness [1/rad]
    mu: float = 1.0  # tyre-road friction coefficient [-]
    c_rr: float = 0.015  # rolling resistance coefficient [-] (assumed)
    cd_a: float = 0.6  # drag coefficient x frontal area [m^2] (assumed)
    max_steer: float = 0.6  # |δ| limit [rad]
    max_steer_rate: float = 1.0  # |δ̇| limit [rad/s]
    steer_tau: float = 0.1  # steering actuator time constant [s]
    max_accel: float = 3.0  # commanded acceleration limit [m/s^2]
    max_brake: float = 6.0  # commanded deceleration limit [m/s^2]
    v_blend_min: float = 3.0  # below: kinematic model [m/s]
    v_blend_max: float = 5.0  # above: dynamic model [m/s]
    kin_tau: float = 0.05  # relaxation of vy, r to kinematic values [s]

    @property
    def wheelbase(self) -> float:
        return self.l_f + self.l_r


class VehicleState(NamedTuple):
    """Continuous state. Position in metres in the world frame."""

    x: jax.Array
    y: jax.Array
    yaw: jax.Array  # ψ
    v_x: jax.Array
    v_y: jax.Array
    yaw_rate: jax.Array  # r
    steer: jax.Array  # δ (actual wheel angle)


class Control(NamedTuple):
    accel: jax.Array  # commanded acceleration a [m/s^2]
    steer: jax.Array  # commanded wheel angle δ_cmd [rad]


def control_from_action(action: jax.Array, p: VehicleParams) -> Control:
    """Map a normalised action in [-1, 1]^2 to physical commands."""
    a, d = jnp.clip(action, -1.0, 1.0)
    accel = jnp.where(a >= 0.0, a * p.max_accel, a * p.max_brake)
    return Control(accel=accel, steer=d * p.max_steer)


def tyre_force(alpha: jax.Array, f_z: float, p: VehicleParams) -> jax.Array:
    """Saturating lateral tyre force: linear (slope cs μ Fz) for small α, ±μ Fz for large α."""
    return p.mu * f_z * jnp.tanh(p.c_s * alpha)


def longitudinal_force(v_x: jax.Array, accel: jax.Array, p: VehicleParams) -> jax.Array:
    """Drive/brake force minus rolling and aerodynamic resistance, limited by friction."""
    f_drive = p.mass * accel
    f_roll = p.c_rr * p.mass * GRAVITY * jnp.tanh(v_x / 0.5)
    f_aero = 0.5 * AIR_DENSITY * p.cd_a * v_x**2
    f_max = p.mu * p.mass * GRAVITY
    return jnp.clip(f_drive - f_roll - f_aero, -f_max, f_max)


def steer_rate(steer: jax.Array, steer_cmd: jax.Array, p: VehicleParams) -> jax.Array:
    """First-order steering actuator with rate limit."""
    return jnp.clip(
        (steer_cmd - steer) / p.steer_tau, -p.max_steer_rate, p.max_steer_rate
    )


def blend_weight(v_x: jax.Array, p: VehicleParams) -> jax.Array:
    """λ = 0 (kinematic) for vx ≤ v_blend_min, 1 (dynamic) for vx ≥ v_blend_max."""
    return jnp.clip((v_x - p.v_blend_min) / (p.v_blend_max - p.v_blend_min), 0.0, 1.0)


def dynamic_derivatives(
    s: VehicleState, f_x: jax.Array, p: VehicleParams
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(v̇x, v̇y, ṙ) of the dynamic single-track model."""
    f_zf = p.mass * GRAVITY * p.l_r / p.wheelbase
    f_zr = p.mass * GRAVITY * p.l_f / p.wheelbase
    # Safe denominator: this branch has zero weight below v_blend_min anyway.
    v_safe = jnp.maximum(s.v_x, 1.0)
    alpha_f = s.steer - jnp.arctan2(s.v_y + p.l_f * s.yaw_rate, v_safe)
    alpha_r = -jnp.arctan2(s.v_y - p.l_r * s.yaw_rate, v_safe)
    f_yf = tyre_force(alpha_f, f_zf, p)
    f_yr = tyre_force(alpha_r, f_zr, p)
    sin_d, cos_d = jnp.sin(s.steer), jnp.cos(s.steer)
    dv_x = (f_x - f_yf * sin_d) / p.mass + s.v_y * s.yaw_rate
    dv_y = (f_yr + f_yf * cos_d) / p.mass - s.v_x * s.yaw_rate
    dr = (p.l_f * f_yf * cos_d - p.l_r * f_yr) / p.inertia_z
    return dv_x, dv_y, dr


def kinematic_derivatives(
    s: VehicleState, f_x: jax.Array, d_steer: jax.Array, p: VehicleParams
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(v̇x, v̇y, ṙ) of the kinematic model written in the dynamic state.

    The no-slip constraint gives r = vx tan δ / L and vy = lr r. Their time
    derivatives are used as feed-forward; a relaxation term pulls vy, r onto
    the constraint manifold after leaving the dynamic regime.
    """
    dv_x = f_x / p.mass
    tan_d = jnp.tan(s.steer)
    r_kin = s.v_x * tan_d / p.wheelbase
    v_y_kin = p.l_r * r_kin
    dr_ff = (dv_x * tan_d + s.v_x * d_steer / jnp.cos(s.steer) ** 2) / p.wheelbase
    dr = dr_ff + (r_kin - s.yaw_rate) / p.kin_tau
    dv_y = p.l_r * dr_ff + (v_y_kin - s.v_y) / p.kin_tau
    return dv_x, dv_y, dr


def euler_step(s: VehicleState, u: Control, h: float, p: VehicleParams) -> VehicleState:
    """One semi-implicit Euler step of length h (velocities first, then pose)."""
    f_x = longitudinal_force(s.v_x, u.accel, p)
    d_steer = steer_rate(s.steer, u.steer, p)
    lam = blend_weight(s.v_x, p)
    dyn = dynamic_derivatives(s, f_x, p)
    kin = kinematic_derivatives(s, f_x, d_steer, p)
    dv_x, dv_y, dr = (lam * a + (1.0 - lam) * b for a, b in zip(dyn, kin, strict=True))

    v_x = jnp.maximum(s.v_x + h * dv_x, 0.0)  # no reverse gear
    stopped = v_x < 1e-3
    v_y = jnp.where(stopped, 0.0, s.v_y + h * dv_y)
    yaw_rate = jnp.where(stopped, 0.0, s.yaw_rate + h * dr)

    yaw = s.yaw + h * yaw_rate
    cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)
    return VehicleState(
        x=s.x + h * (v_x * cos_y - v_y * sin_y),
        y=s.y + h * (v_x * sin_y + v_y * cos_y),
        yaw=jnp.arctan2(sin_y, cos_y),
        v_x=v_x,
        v_y=v_y,
        yaw_rate=yaw_rate,
        steer=jnp.clip(s.steer + h * d_steer, -p.max_steer, p.max_steer),
    )


def simulate(
    s: VehicleState, u: Control, duration: float, num_steps: int, p: VehicleParams
) -> VehicleState:
    """Hold control u for ``duration`` seconds using ``num_steps`` Euler sub-steps."""
    h = duration / num_steps
    return jax.lax.fori_loop(0, num_steps, lambda _, st: euler_step(st, u, h, p), s)


def lateral_acceleration(s: VehicleState) -> jax.Array:
    """Centripetal acceleration a_y ≈ vx r [m/s^2] (steady-state approximation)."""
    return s.v_x * s.yaw_rate
