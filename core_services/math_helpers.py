"""
Vector math utilities used throughout the topic-evolution pipeline.

These helpers intentionally stay:
- **Stateless** and **pure** (easy to unit-test).
- **Numerically safe** (small epsilons to avoid div-by-zero).
- **Generic** (no dependency on a specific embedding model).

Conceptual notes:
- Embeddings are treated as points in a high-dimensional Euclidean space.
- Cosine similarity is the primary affinity measure for clustering/matching.
- We use two flavors of averaging:
  (1) **Weighted incremental mean** for the long-term topic centroid (O(1) update).
  (2) **Exponential moving average (EMA)** for short-term smoothing (recency bias).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import math, random
random.seed(7)  # deterministic toy behavior when this module is imported in quick experiments

Vector = List[float]


def dot(a: Vector, b: Vector) -> float:
    """
    Dot product  ⟨a, b⟩ = Σ_i a_i * b_i

    Theory:
      - The dot product measures alignment of two vectors scaled by their magnitudes.
      - It’s the building block for cosine similarity (cos = dot / (||a||·||b||)).

    Usage:
      - Used by `cos()` and `norm()`; unit tested for correctness.
    """
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """
    Euclidean (L2) norm  ||a|| = sqrt(⟨a, a⟩)

    Theory:
      - Magnitude (length) of a vector.
      - Needed to normalize for cosine similarity.
    Numerical safety:
      - Lower-bounds the squared norm by 1e-12 to avoid zero-division downstream.
    """
    return math.sqrt(max(1e-12, dot(a, a)))


def cos(a: Vector, b: Vector) -> float:
    """
    Cosine similarity  cos(a, b) = ⟨a, b⟩ / (||a|| · ||b||)

    Theory:
      - Measures *directional* similarity, independent of magnitude.
      - Range is [-1, 1]; with most sentence/semantic embeddings you’ll see [~0, 1].
      - Preferred in high-D embedding spaces where scale can vary.

    Usage:
      - Cluster matching (snapshot -> state).
      - Separation checks (cosine to parent centroid).
      - Higher is “more similar”.
    """
    return dot(a, b) / (norm(a) * norm(b))


def add(a: Vector, b: Vector) -> Vector:
    """
    Element-wise vector addition: (a + b)_i = a_i + b_i

    Usage:
      - Utility for building simple accumulators (used by `mean()`).
    """
    return [x + y for x, y in zip(a, b)]


def scale(a: Vector, s: float) -> Vector:
    """
    Scalar multiplication: (s · a)_i = s * a_i

    Usage:
      - Normalize accumulators into means.
      - Apply learning-rate-like scaling.
    """
    return [x * s for x in a]


def mean(vs: List[Vector]) -> Vector:
    """
    Plain (unweighted) arithmetic mean of a list of vectors.

    Theory:
      - c = (1/N) Σ v_i
      - Assumes all vectors have the same dimensionality.

    Caveats:
      - For *online* updates with weights (authority/time), use `weighted_incremental_mean`.
      - For large lists, prefer a stable/streaming approach to reduce accumulation error.
    """
    acc = [0.0] * len(vs[0])
    # Imperative accumulate (explicit for readability over clever one-liners)
    for v in vs:
        acc = add(acc, v)
    return scale(acc, 1.0 / len(vs))


def weighted_incremental_mean(
    c_prev: Optional[Vector], W_prev: float, e: Vector, w: float
) -> Tuple[Vector, float]:
    """
    Online (one-pass) update of a **weighted mean** centroid.

    Problem:
      - Maintain c = (Σ w_i * e_i) / (Σ w_i) as new (e, w) arrive, without storing history.

    Update:
      - Given previous (c_prev, W_prev) and new (e, w):
        W = W_prev + w
        c_new = (c_prev * W_prev + e * w) / W

    Returns:
      - (c_new, W) where W is the updated total weight.

    Why this matters:
      - **O(1)** memory & time per doc → perfect for long-term topic centroids.
      - w can encode *authority*, *recency decay*, and *sample_weight* (duplicate aggregation).

    Edge cases:
      - If no prior mass (c_prev is None or W_prev <= 0), we bootstrap with the new point.
    """
    if c_prev is None or W_prev <= 0:
        return (e[:], w)
    W = W_prev + w
    c_new = [(ci * W_prev + ei * w) / W for ci, ei in zip(c_prev, e)]
    return (c_new, W)


def ema_update_vec(v_ema: Optional[Vector], v_now: Vector, beta: float) -> Vector:
    """
    Exponential Moving Average (EMA) update for vectors.

    Formula (per dimension):
      v_ema ← (1 - β) · v_ema + β · v_now

    Theory:
      - EMA gives **recency-weighted** smoothing with an exponential kernel.
      - Effective half-life h (in steps) relates to β by: β = 1 - 2^(-1/h).
        (Higher β → more reactive; lower β → smoother.)

    Usage:
      - Smooth cluster centroids across ticks (stabilize matching/persistence).
      - Optional: smooth topic short-term centroid (drift lens).
    """
    if v_ema is None:
        return v_now[:]
    return [(1.0 - beta) * e + beta * n for e, n in zip(v_ema, v_now)]


def wmean(vecs: List[Vector], weights: List[float]) -> Tuple[Vector, float]:
    """
    Weighted mean over a batch of vectors (batch form).

    Inputs:
      - vecs: list of vectors (same dimension).
      - weights: non-negative weights (e.g., authority × sample_weight).

    Returns:
      - (mean_vec, W) where mean_vec = (Σ w_i v_i) / (Σ w_i) and W = Σ w_i.

    Theory & when to use:
      - Used inside batch algorithms (e.g., k-means centroid recomputation) when
        each example has a different *mass* (dedup aggregation) or importance.
      - Complements `weighted_incremental_mean`, which updates point-by-point.

    Numerical notes:
      - Guards division by zero with a tiny epsilon (1e-12) if all weights are zero.
    """
    dim = len(vecs[0])
    acc = [0.0] * dim
    W = 0.0
    for v, w in zip(vecs, weights):
        W += w
        for i in range(dim):
            acc[i] += v[i] * w
    return [x / max(W, 1e-12) for x in acc], W
