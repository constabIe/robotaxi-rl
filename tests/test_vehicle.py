"""Physical plausibility of the single-track model against closed-form results."""

import math

import jax
import jax.numpy as jnp
import pytest

from env.vehicle import GRAVITY, Control, VehicleParams, VehicleState, simulate

simulate_jit = jax.jit(simulate, static_argnums=(2, 3, 4))


def state(v: float = 0.0) -> VehicleState:
    z = jnp.asarray(0.0)
    return VehicleState(z, z, z, jnp.asarray(v), z, z, z)


def control(a: float, d: float = 0.0) -> Control:
    return Control(jnp.asarray(a), jnp.asarray(d))


def test_standstill_is_equilibrium():
    s = simulate_jit(state(), control(0.0, 0.3), 5.0, 1200, VehicleParams())
    assert float(jnp.hypot(s.x, s.y)) < 1e-6
    assert all(bool(jnp.isfinite(v)) for v in s)


def test_straight_acceleration():
    p = VehicleParams()
    s = simulate_jit(state(), control(p.max_accel), 2.0, 480, p)
    # v = a t minus small rolling/aero losses
    assert 5.5 < float(s.v_x) < p.max_accel * 2.0


@pytest.mark.parametrize("mu", [1.0, 0.5, 0.1])
def test_braking_distance(mu):
    p = VehicleParams(mu=mu)
    v0 = 10.0
    s = simulate_jit(state(v0), control(-p.max_brake), 15.0, 3600, p)
    decel = min(p.max_brake, mu * GRAVITY)
    expected = v0**2 / (2 * decel)
    assert float(s.v_x) == 0.0
    assert float(s.x) == pytest.approx(expected, rel=0.1)


def test_kinematic_turning_radius():
    p = VehicleParams()
    delta = 0.3
    s = simulate_jit(state(2.0), control(0.15, delta), 5.0, 1200, p)
    radius = float(s.v_x / s.yaw_rate)
    assert radius == pytest.approx(p.wheelbase / math.tan(delta), rel=0.02)


def test_low_friction_understeer():
    """At speed with full lock, a slippery road gives a much larger turning radius."""
    radii = {}
    for mu in (1.0, 0.1):
        s = simulate_jit(state(12.0), control(0.0, 0.6), 2.0, 480, VehicleParams(mu=mu))
        radii[mu] = float(s.v_x / jnp.maximum(jnp.abs(s.yaw_rate), 1e-6))
    assert radii[0.1] > 5 * radii[1.0]


def test_blend_is_continuous():
    """Crossing the kinematic/dynamic blending band does not create jumps in v_y or r."""
    p = VehicleParams()

    def body(s, _):
        s = simulate(s, control(1.0, 0.1), 1 / 60, 4, p)
        return s, (s.v_y, s.yaw_rate)

    _, (v_y, r) = jax.lax.scan(body, state(2.0), None, length=180)
    assert float(jnp.max(jnp.abs(jnp.diff(v_y)))) < 0.05
    assert float(jnp.max(jnp.abs(jnp.diff(r)))) < 0.05
