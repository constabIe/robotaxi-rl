"""Robotaxi environment: BaseEnv with single-track vehicle dynamics.

Interface (same as ``BaseEnv``; functions are called on the class):

    params, init_state = RobotaxiEnv.init_params(key, map_id=1, max_steps=450, path_length=100)
    obs, state = RobotaxiEnv.reset(key, params, init_state)
    obs, state, reward, done, info = RobotaxiEnv.step(key, state, action, params)

* action  u = (a, δ) ∈ [-1, 1]²: normalised acceleration and steering angle command;
* state   BaseEnvState + (ψ, v_x, v_y, r, δ, u_{t-1});
* obs     ``RobotaxiObservation``; ``obs.as_vector()`` is the policy input.

Positions stay in map cells (as in BaseEnv, needed by the renderer and the
planner); the vehicle model works in metres, scaled by ``meters_per_cell``.
``BaseEnv.step``/``reset`` are re-implemented because they call
``BaseEnv.get_observation`` directly, and they evaluate termination on the
pre-step state; the base helpers (planner, ray generation, ray-disc
intersection, goal test, obstacle motion) are reused. Collisions use the car's
own radius instead of BaseEnv's 0.5-cell (2.5 m) agent disc.
"""

import dataclasses
import math
from functools import partial
from typing import Any

import chex
import equinox
import jax
from jax import numpy as jnp

from env.base import FOV, NUM_RAY_SENSORS, BaseEnv, BaseEnvParams, BaseEnvState
from env.rewards import RewardInputs, RewardWeights, reward_terms, total_reward
from env.vehicle import (
    VehicleParams,
    VehicleState,
    control_from_action,
    lateral_acceleration,
    simulate,
)
from utils.geometry import lookahead_points, path_frame, to_body


class RobotaxiParams(BaseEnvParams):
    vehicle: VehicleParams
    reward: RewardWeights
    meters_per_cell: float  # map scale [m / cell]
    action_repeat: int  # physics frames per agent decision
    substeps: int  # Euler sub-steps per physics frame
    replan_every: int  # environment steps between path re-planning
    num_lookahead: int  # path points in the observation
    lookahead_stride: int  # vertices between observed path points
    collision_radius: (
        float  # centre distance at which the car touches an obstacle [cells]
    )


class RobotaxiState(BaseEnvState):
    yaw: chex.Array  # ψ [rad]
    v_x: chex.Array  # [m/s]
    v_y: chex.Array  # [m/s]
    yaw_rate: chex.Array  # r [rad/s]
    steer: chex.Array  # δ [rad]
    prev_action: chex.Array  # u_{t-1} (2,)


class RobotaxiObservation(equinox.Module):
    """All features are expressed in the body frame and scaled to O(1)."""

    ego: chex.Array  # (v_x/v_lim, v_y/v_lim, r, δ/δ_max)
    prev_action: chex.Array  # u_{t-1}
    path_error: chex.Array  # (e_lat/2, cos e_ψ, sin e_ψ)
    lookahead: chex.Array  # (K, 2) next path points / perception radius
    goal: chex.Array  # (‖g‖/diag, ĝ_x, ĝ_y)
    ray_distances: chex.Array  # (N,) / perception radius
    time_left: chex.Array  # (1,) fraction of the episode left
    collision_rays: (
        chex.Array
    )  # (N, 2) world-frame ray vectors, used by the renderer only

    def as_vector(self) -> chex.Array:
        parts = (
            self.ego,
            self.prev_action,
            self.path_error,
            self.lookahead,
            self.goal,
            self.ray_distances,
            self.time_left,
        )
        batch = self.ego.shape[
            :-1
        ]  # works for a single observation and for vmapped batches
        return jnp.concatenate([x.reshape(*batch, -1) for x in parts], axis=-1)


def _heading_towards_path(pos: chex.Array, path: chex.Array) -> chex.Array:
    """Yaw pointing from ``pos`` to the second path vertex that is not ``pos`` itself."""
    far = jnp.linalg.norm(path - pos, axis=-1) > 0.5
    first = jnp.argmax(far)
    target = path[jnp.minimum(first + 1, path.shape[0] - 1)]
    d = target - pos
    return jnp.arctan2(d[1], d[0])


def _vehicle_state(s: RobotaxiState, scale: float) -> VehicleState:
    return VehicleState(
        x=s.agent_pos[0] * scale,
        y=s.agent_pos[1] * scale,
        yaw=s.yaw,
        v_x=s.v_x,
        v_y=s.v_y,
        yaw_rate=s.yaw_rate,
        steer=s.steer,
    )


OBSTACLE_RADIUS = 0.5  # obstacles are discs inscribed in their cell [cells]


def cast_rays(
    pos: chex.Array, forward: chex.Array, obstacles: chex.Array, p: RobotaxiParams
) -> tuple[chex.Array, chex.Array]:
    """Unit ray directions (N, 2) and distances to the first contact (N,), capped by the perception radius.

    Distances are measured to the configuration-space obstacle (disc of radius
    ``collision_radius``), so a distance of 0 means contact.
    """
    rays = BaseEnv._generate_rays(forward, n_rays=p.num_ray_sensors, fov=p.agent_FOV)
    dists = BaseEnv._ray_circle_intersections(pos, rays, obstacles, p.collision_radius)
    return rays, jnp.minimum(dists, p.perception_radius)


def _collides(pos: chex.Array, obstacles: chex.Array, radius: float) -> chex.Array:
    return jnp.any(jnp.linalg.norm(obstacles - pos, axis=-1) < radius)


def _off_map(pos: chex.Array, map_height_width: tuple[int, int]) -> chex.Array:
    h, w = map_height_width
    return jnp.any(pos < -0.5) | (pos[0] > w - 0.5) | (pos[1] > h - 0.5)


class RobotaxiEnv(BaseEnv):
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5.0,
        num_ray_sensors: int = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: int = 60,
        *,
        vehicle: VehicleParams | None = None,
        reward: RewardWeights | None = None,
        meters_per_cell: float = 8.0,
        action_repeat: int = 4,
        substeps: int = 4,
        num_lookahead: int = 5,
        lookahead_stride: int = 2,
        car_radius_m: float = 1.0,
    ) -> tuple[RobotaxiParams, RobotaxiState]:
        base_params, base_state = BaseEnv.init_params(
            key,
            map_id,
            max_steps,
            path_length,
            discretization_scale,
            perception_radius,
            num_ray_sensors,
            fov,
            fps,
        )
        base_fields = {
            f.name: getattr(base_params, f.name)
            for f in dataclasses.fields(base_params)
        }
        # step_size is the duration of one environment step; BaseEnv uses it to move obstacles.
        base_fields["step_size"] = action_repeat / fps
        params = RobotaxiParams(
            **base_fields,
            vehicle=vehicle if vehicle is not None else VehicleParams(),
            reward=reward if reward is not None else RewardWeights(),
            meters_per_cell=float(meters_per_cell),
            action_repeat=int(action_repeat),
            substeps=int(substeps),
            replan_every=max(1, round(fps / action_repeat)),
            num_lookahead=int(num_lookahead),
            lookahead_stride=int(lookahead_stride),
            collision_radius=OBSTACLE_RADIUS + car_radius_m / meters_per_cell,
        )

        yaw = _heading_towards_path(base_state.agent_pos, base_state.path_array)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        base_state_fields = {
            f.name: getattr(base_state, f.name) for f in dataclasses.fields(base_state)
        }
        base_state_fields["agent_forward_dir"] = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)])
        init_state = RobotaxiState(
            **base_state_fields,
            yaw=yaw,
            v_x=zero,
            v_y=zero,
            yaw_rate=zero,
            steer=zero,
            prev_action=jnp.zeros(2, dtype=jnp.float32),
        )
        return params, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: RobotaxiParams, init_state: RobotaxiState | None
    ) -> tuple[RobotaxiObservation, RobotaxiState]:
        if init_state is None:
            raise ValueError("RobotaxiEnv.reset needs init_state from init_params")
        return RobotaxiEnv.get_observation(init_state, env_params), init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RobotaxiState,
        action: chex.Array,
        env_params: RobotaxiParams,
    ) -> tuple[
        RobotaxiObservation, RobotaxiState, chex.Array, chex.Array, dict[str, Any]
    ]:
        p, scale = env_params, env_params.meters_per_cell
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)

        # 1. Vehicle dynamics over one environment step (action held constant).
        vs = simulate(
            _vehicle_state(env_state, scale),
            control_from_action(action, p.vehicle),
            p.step_size,
            p.action_repeat * p.substeps,
            p.vehicle,
        )
        pos = jnp.stack([vs.x, vs.y]) / scale

        # 2. World: moving obstacles, time, periodic re-planning.
        kinematic = BaseEnv._move_kinematic_obstacles(env_state, p)
        obstacles = jnp.concatenate([env_state.static_obstacles, kinematic], axis=0)
        time = env_state.time + 1
        path = jax.lax.cond(
            jnp.mod(time, p.replan_every) == 0,
            lambda: BaseEnv._find_path(pos, env_state.goal_pos, obstacles, p),
            lambda: env_state.path_array,
        )

        new_state = RobotaxiState(
            time=time,
            goal_pos=env_state.goal_pos,
            agent_pos=pos,
            agent_forward_dir=jnp.stack([jnp.cos(vs.yaw), jnp.sin(vs.yaw)]),
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path,
            yaw=vs.yaw,
            v_x=vs.v_x,
            v_y=vs.v_y,
            yaw_rate=vs.yaw_rate,
            steer=vs.steer,
            prev_action=action,
        )

        # 3. Termination, evaluated on the new state.
        goal = BaseEnv._check_goal(pos, env_state.goal_pos)
        collision = _collides(pos, obstacles, p.collision_radius)
        off_map = _off_map(pos, p.map_height_width)
        terminated = goal | collision | off_map
        truncated = (time >= p.max_steps_in_episode) & ~terminated
        done = terminated | truncated

        # 4. Reward. Both potentials use the same (possibly re-planned) path.
        before = path_frame(env_state.agent_pos, path)
        after = path_frame(pos, path)
        terms = reward_terms(
            RewardInputs(
                remaining_before=before.remaining * scale,
                remaining_after=after.remaining * scale,
                lateral=after.lateral * scale,
                v_x=vs.v_x,
                lat_accel=lateral_acceleration(vs),
                action=action,
                prev_action=env_state.prev_action,
                goal=goal,
                crash=collision | off_map,
            ),
            p.reward,
        )
        reward = total_reward(terms)

        obs = RobotaxiEnv.get_observation(new_state, p)
        info = {
            "time": time,
            "distance_to_path": jnp.abs(after.lateral),
            "direction_of_path": after.tangent,
            "speed_kmh": 3.6 * jnp.hypot(vs.v_x, vs.v_y),
            "remaining_m": after.remaining * scale,
            "goal": goal,
            "collision": collision | off_map,
            "terminated": terminated,
            "truncated": truncated,
            **terms,
        }
        return obs, new_state, reward, done, info

    def get_observation(
        env_state: RobotaxiState, env_params: RobotaxiParams
    ) -> RobotaxiObservation:
        p = env_params
        pos, yaw = env_state.agent_pos, env_state.yaw
        v_lim = p.reward.speed_limit

        frame = path_frame(pos, env_state.path_array)
        tangent_body = to_body(frame.tangent, yaw)  # = (cos e_ψ, sin e_ψ)

        ahead = lookahead_points(
            env_state.path_array, frame.segment, p.num_lookahead, p.lookahead_stride
        )
        ahead_body = to_body(ahead - pos, yaw) / p.perception_radius

        to_goal = to_body(env_state.goal_pos - pos, yaw)
        goal_dist = jnp.linalg.norm(to_goal)
        diag = math.hypot(*p.map_height_width)
        goal_feat = jnp.concatenate(
            [jnp.atleast_1d(goal_dist / diag), to_goal / jnp.maximum(goal_dist, 1e-6)]
        )

        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays, dists = cast_rays(pos, env_state.agent_forward_dir, obstacles, p)

        return RobotaxiObservation(
            ego=jnp.stack(
                [
                    env_state.v_x / v_lim,
                    env_state.v_y / v_lim,
                    env_state.yaw_rate,
                    env_state.steer / p.vehicle.max_steer,
                ]
            ),
            prev_action=env_state.prev_action,
            path_error=jnp.stack(
                [frame.lateral / 2.0, tangent_body[0], tangent_body[1]]
            ),
            lookahead=ahead_body,
            goal=goal_feat,
            ray_distances=dists / p.perception_radius,
            time_left=jnp.atleast_1d(1.0 - env_state.time / p.max_steps_in_episode),
            collision_rays=rays * dists[:, None],
        )
