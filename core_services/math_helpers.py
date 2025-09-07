from __future__ import annotations
from typing import List, Tuple, Optional
import math, random
random.seed(7)


Vector = List[float]
def dot(a: Vector, b: Vector) -> float: 
    return sum(x*y for x, y in zip(a, b))

def norm(a: Vector) -> float: 
    return math.sqrt(max(1e-12, dot(a, a)))

def cos(a: Vector, b: Vector) -> float: 
    return dot(a, b) / (norm(a) * norm(b))

def add(a: Vector, b: Vector) -> Vector: 
    return [x+y for x,y in zip(a,b)]

def scale(a: Vector, s: float) -> Vector: 
    return [x*s for x in a]

def mean(vs: List[Vector]) -> Vector:
    acc = [0.0]*len(vs[0]);  
    [acc:=add(acc,v) for v in vs];  
    return scale(acc, 1.0/len(vs))

def weighted_incremental_mean(c_prev: Optional[Vector], W_prev: float, e: Vector, w: float) -> Tuple[Vector, float]:
    if c_prev is None or W_prev <= 0: 
        return (e[:], w)
    W = W_prev + w
    c_new = [ (ci*W_prev + ei*w)/W for ci, ei in zip(c_prev, e) ]
    return (c_new, W)

def ema_update_vec(v_ema: Optional[Vector], v_now: Vector, beta: float) -> Vector:
    if v_ema is None: return v_now[:]
    return [ (1.0-beta)*e + beta*n for e,n in zip(v_ema, v_now) ]


def wmean(vecs, weights):
    dim = len(vecs[0])
    acc = [0.0]*dim
    W = 0.0
    for v, w in zip(vecs, weights):
        W += w
        for i in range(dim): acc[i] += v[i]*w
    return [x / max(W, 1e-12) for x in acc], W
