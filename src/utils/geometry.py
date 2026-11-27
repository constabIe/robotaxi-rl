"""Pure geometric helpers for paths and frames (2D, jit/vmap friendly)."""

from typing import NamedTuple

import jax
from jax import numpy as jnp


def rotation(yaw: jax.Array) -> jax.Array:
    """R(ψ) mapping body-frame vectors to the world frame."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array([[c, -s], [s, c]])


def to_body(vec_world: jax.Array, yaw: jax.Array) -> jax.Array:
    """Express world-frame vector(s) (..., 2) in the body frame: Rᵀ(ψ) v."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    x, y = vec_world[..., 0], vec_world[..., 1]
    return jnp.stack([c * x + s * y, -s * x + c * y], axis=-1)


def cross2(a: jax.Array, b: jax.Array) -> jax.Array:
    """z-component of the 2D cross product a × b."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class PathFrame(NamedTuple):
    """Position of a point relative to a polyline path."""

    remaining: jax.Array  # distance to the path + arc length from the projection to the end
    lateral: jax.Array  # signed distance to the path (+ left of the path)
    tangent: jax.Array  # unit tangent of the closest segment (2,)
    segment: jax.Array  # index of the closest segment
    arc: jax.Array  # arc length from the path start to the projection


def path_frame(point: jax.Array, path: jax.Array) -> PathFrame:
    """Project ``point`` onto the polyline ``path`` (N, 2).

    The projection is taken on the closest *segment*, which makes the remaining
    length continuous when the closest vertex changes. The distance from the
    point to its projection is added, so that D also decreases while the agent
    approaches a path that starts ahead of it. Degenerate
    (zero-length) segments, e.g. the padding at the goal, are skipped.
    """
    a, b = path[:-1], path[1:]
    seg = b - a
    seg_len = jnp.linalg.norm(seg, axis=-1)
    valid = seg_len > 1e-6
    safe_len = jnp.where(valid, seg_len, 1.0)
    t = jnp.clip(jnp.sum((point - a) * seg, axis=-1) / safe_len**2, 0.0, 1.0)
    proj = a + t[:, None] * seg
    dist = jnp.where(valid, jnp.linalg.norm(point - proj, axis=-1), jnp.inf)

    # All segments degenerate (agent already at the goal cell): distance to the end.
    any_valid = jnp.any(valid)
    k = jnp.argmin(dist)
    tail = jnp.cumsum(jnp.where(valid, seg_len, 0.0)[::-1])[::-1]  # length from segment k start
    remaining = jnp.where(
        any_valid,
        tail[k] - t[k] * seg_len[k] + dist[k],
        jnp.linalg.norm(point - path[-1]),
    )
    tangent = jnp.where(any_valid, seg[k] / safe_len[k], jnp.array([1.0, 0.0]))
    lateral = jnp.where(any_valid, cross2(tangent, point - proj[k]), 0.0)
    arc = jnp.sum(jnp.where(valid, seg_len, 0.0)) - tail[k] + t[k] * seg_len[k]
    return PathFrame(remaining=remaining, lateral=lateral, tangent=tangent, segment=k, arc=arc)


def point_at_arc(path: jax.Array, arc: jax.Array) -> jax.Array:
    """Point of the polyline at arc length ``arc`` from its start (clamped to the ends)."""
    seg = path[1:] - path[:-1]
    seg_len = jnp.linalg.norm(seg, axis=-1)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_len)])
    arc = jnp.clip(arc, 0.0, cum[-1])
    j = jnp.clip(jnp.searchsorted(cum, arc, side="right") - 1, 0, seg.shape[0] - 1)
    frac = jnp.where(seg_len[j] > 1e-6, (arc - cum[j]) / jnp.maximum(seg_len[j], 1e-6), 0.0)
    return path[j] + jnp.clip(frac, 0.0, 1.0) * seg[j]


def lookahead_points(path: jax.Array, segment: jax.Array, num: int, stride: int) -> jax.Array:
    """``num`` path vertices after ``segment`` taken every ``stride`` vertices (clipped at the end)."""
    idx = jnp.clip(segment + 1 + stride * jnp.arange(num), 0, path.shape[0] - 1)
    return path[idx]
