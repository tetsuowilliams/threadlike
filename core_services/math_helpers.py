"""
math_helpers.py (weight-free)

Minimal vector ops for cosine-space clustering and smoothing.
- All ops are pure & numerically safe (tiny eps where needed).
- No sample_weight anywhere.
"""

from __future__ import annotations
from typing import List, Optional
import math

Vector = List[float]
EPS = 1e-12


# ---------- basic ops ----------
def dot(a: Vector, b: Vector) -> float:
    """Dot product ⟨a,b⟩."""
    return sum(x * y for x, y in zip(a, b))

def norm(a: Vector) -> float:
    """L2 norm ||a||."""
    return math.sqrt(max(EPS, dot(a, a)))

def cos(a: Vector, b: Vector) -> float:
    """Cosine similarity = ⟨a,b⟩ / (||a||·||b||)."""
    return dot(a, b) / (norm(a) * norm(b))

def add(a: Vector, b: Vector) -> Vector:
    """Elementwise a + b."""
    return [x + y for x, y in zip(a, b)]

def scale(a: Vector, s: float) -> Vector:
    """Elementwise s * a."""
    return [x * s for x in a]


# ---------- means & centroids ----------
def mean(vs: List[Vector]) -> Vector:
    """
    Plain (unweighted) arithmetic mean of vectors.
    Assumes vs is non-empty and all same dim.
    """
    acc = [0.0] * len(vs[0])
    for v in vs:
        for i, x in enumerate(v):
            acc[i] += x
    n = max(1, len(vs))
    return [x / n for x in acc]

def incremental_mean(c_prev: Optional[Vector], n_prev: int, e: Vector) -> tuple[Vector, int]:
    """
    O(1) update of an unweighted mean.
      Given previous centroid c_prev over n_prev items,
      new centroid after adding e is:
        c_new = (c_prev * n_prev + e) / (n_prev + 1)
    Returns (c_new, n_new).
    """
    if c_prev is None or n_prev <= 0:
        return (e[:], 1)
    n_new = n_prev + 1
    c_new = [(ci * n_prev + ei) / n_new for ci, ei in zip(c_prev, e)]
    return (c_new, n_new)


# ---------- smoothing ----------
def ema_update_vec(v_ema: Optional[Vector], v_now: Vector, beta: float) -> Vector:
    """
    Exponential Moving Average for vectors:
      v_ema <- (1 - beta) * v_ema + beta * v_now
    If v_ema is None, bootstrap with v_now.
    """
    if v_ema is None:
        return v_now[:]
    return [(1.0 - beta) * e + beta * n for e, n in zip(v_ema, v_now)]


# ---------- cosine-friendly helpers ----------
def l2_normalize(v: Vector) -> Vector:
    """Return unit-L2 version of v."""
    n = norm(v)
    return [x / n for x in v]

def l2_normalize_rows(M: List[Vector]) -> List[Vector]:
    """Row-wise L2 normalize a list/matrix of vectors."""
    out: List[Vector] = []
    for v in M:
        n = math.sqrt(max(EPS, dot(v, v)))
        out.append([x / n for x in v])
    return out

def centroid_unit(vs: List[Vector]) -> Vector:
    """
    Mean centroid, then L2-normalize (good for cosine space).
    """
    c = mean(vs)
    return l2_normalize(c)

def cohesion_mean_cos(X: List[Vector], C: Vector) -> float:
    """
    Average cosine(doc, C) over X. Assumes C is unit (use centroid_unit).
    """
    # If X rows are also unit, cos = dot; but we compute safely anyway.
    s = 0.0
    for v in X:
        s += cos(v, C)
    return s / max(1, len(X))
