#!/usr/bin/env python3
# gen_tick.py
#
# Generate deterministic JSON "tick" files for the temporal topic-branching testbench.
#
# Each tick_NNN.json contains an array of raw "hits" your retriever would have returned.
# You can feed these into your FixtureSearch/FixtureFetch or preload path.
#
# Usage examples:
#   # Default JS runtimes scenario, 8 ticks → tests/corpus/
#   python gen_tick.py --out tests/corpus --ticks 8 --seed 42
#
#   # Add some near-duplicate pressure (same text, different URLs) within ticks
#   python gen_tick.py --out tests/corpus --ticks 8 --seed 42 --dupe-rate 0.2
#
#   # Use a custom plan file describing per-tick counts per bucket
#   python gen_tick.py --out tests/corpus --plan plan.json
#
# Plan file format (JSON):
# [
#   {"Parent": 4, "Node": 2, "Deno": 0, "Bun": 0, "Noise": 0},
#   {"Parent": 3, "Node": 3, "Deno": 0, "Bun": 0, "Noise": 0},
#   {"Parent": 2, "Node": 2, "Deno": 2, "Bun": 0, "Noise": 0},
#   ...
# ]
#
# Notes:
# - We normalize/lower text when hashing in your pipeline; duplicates here help test dedup/weights.
# - Authority/type/domain are synthesized per bucket; tweak maps below as needed.

from __future__ import annotations
import argparse, random, json, time, hashlib
from pathlib import Path
from typing import Dict, List

# -------------------------
# Vocab buckets (JS runtimes)
# -------------------------
PARENT = [
    "javascript", "runtime", "esm", "cjs", "http server", "tooling",
    "event loop", "workers", "ts", "modules"
]
NODE = [
    "npm", "CommonJS", "require", "Express", "NestJS", "libuv",
    "cluster", "worker_threads", "package.json", "LTS"
]
DENO = [
    "TypeScript-first", "permissions", "deno.json", "std library",
    "Web APIs", "import URL", "Deno KV", "fetch", "bundled tooling"
]
BUN = [
    "bun install", "bundler", "test runner", "HMR", "Zig",
    "native loader", "fast startup", "hot reload"
]
NOISE = [
    "bootcamp", "tutorial", "hiring", "course", "newsletter",
    "learn quickly", "interview questions"
]

BUCKETS: Dict[str, List[str]] = {
    "Parent": PARENT, "Node": NODE, "Deno": DENO, "Bun": BUN, "Noise": NOISE
}

# Authority priors by bucket (0..1). Docs > blogs > hype/noise.
AUTHORITY = {
    "Parent": 0.70,
    "Node":   0.70,
    "Deno":   0.85,
    "Bun":    0.65,
    "Noise":  0.30,
}

# Type priors (purely illustrative)
DTYPE = {
    "Parent": "blog",
    "Node":   "blog",
    "Deno":   "docs",
    "Bun":    "news",
    "Noise":  "forum",
}

# Domain priors
DOMAIN = {
    "Parent": "example.dev",
    "Node":   "nodejs.example",
    "Deno":   "deno.land",
    "Bun":    "bun.sh",
    "Noise":  "random.forum",
}

# -------------------------
# Helpers
# -------------------------
def norm_text(s: str) -> str:
    """Minimal normalization so tiny diffs don't explode hashes downstream."""
    return " ".join(s.lower().split())

def text_hash(s: str) -> str:
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()

def synth_sentence(bucket: str, vocab: List[str], n_tokens: int = 12, rng: random.Random = random) -> str:
    """Create a small bag-of-words sentence from the bucket vocab."""
    tokens = rng.choices(vocab, k=n_tokens)
    # Light template to look like a sentence
    return f"{bucket} topic: " + " ".join(tokens)

def synth_record(bucket: str, idx: int, base_ts: int, rng: random.Random) -> Dict:
    """Build one raw hit record for a bucket."""
    vocab = BUCKETS[bucket]
    
    # Sample sentence & fields
    text = synth_sentence(bucket, vocab, n_tokens=rng.randint(10, 16), rng=rng)
    title = f"{bucket} doc {idx}"
    domain = DOMAIN[bucket]
    dtype = DTYPE[bucket]
    auth = AUTHORITY[bucket] + rng.uniform(-0.05, 0.05)
    auth = max(0.0, min(1.0, auth))

    # Jitter timestamp inside the tick
    ts = base_ts + rng.randint(0, 3600)

    # URL unique-ish per record
    url = f"https://{domain}/{bucket.lower()}/{idx}"
    return {
        "url": url,
        "title": title,
        "domain": domain,
        "type": dtype,
        "ts": ts,
        "authority": auth,
        "text": text,
        # hash is optional; your loader can derive if missing
        # "hash": text_hash(text),
        "arm_id": f"seed:{bucket}"
    }

def make_dupe_of(rec: Dict, dupe_idx: int) -> Dict:
    """Create a duplicate entry (same text) but a different URL to test within-tick dedup."""
    rec2 = dict(rec)
    # different URL path to simulate alternate surfacing
    rec2["url"] = rec["url"] + f"?alt={dupe_idx}"
    return rec2

def default_plan(ticks: int) -> List[Dict[str, int]]:
    """
    Build a simple 8-tick drift plan if no custom plan provided:
      t0-1: Parent+Node
      t2-3: add Deno
      t4-5: add Bun
      t6-7: inject Noise
    If ticks != 8, we stretch/compress proportionally.
    """
    base = [
        {"Parent": 2, "Node": 4, "Deno": 0, "Bun": 0, "Noise": 0},
        {"Parent": 2, "Node": 4, "Deno": 0, "Bun": 0, "Noise": 0},
        {"Parent": 2, "Node": 2, "Deno": 2, "Bun": 0, "Noise": 0},
        {"Parent": 1, "Node": 2, "Deno": 3, "Bun": 0, "Noise": 0},
        {"Parent": 1, "Node": 1, "Deno": 3, "Bun": 2, "Noise": 0},
        {"Parent": 0, "Node": 1, "Deno": 2, "Bun": 4, "Noise": 0},
        {"Parent": 0, "Node": 1, "Deno": 2, "Bun": 3, "Noise": 2},
        {"Parent": 0, "Node": 1, "Deno": 1, "Bun": 3, "Noise": 3},
    ]
    if ticks == 8:
        return base
    # Stretch/compress: map each desired tick to nearest base index
    plan = []
    for i in range(ticks):
        j = round(i * (len(base) - 1) / max(1, (ticks - 1)))
        plan.append(base[j])
    return plan

# -------------------------
# Main generation
# -------------------------
def generate_ticks(out_dir: Path, ticks: int, plan: List[Dict[str, int]], dupe_rate: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Start timestamps near "now", step per tick (1 day spacing by default)
    day = 24 * 3600
    t0 = int(time.time())

    for t in range(ticks):
        tick_ts = t0 + t * day
        # Build records per bucket according to plan
        plan_row = plan[t] if t < len(plan) else plan[-1]
        records: List[Dict] = []

        # Counter per bucket for URL uniqueness
        counters = {k: 0 for k in BUCKETS.keys()}

        for bucket, count in plan_row.items():
            for _ in range(count):
                idx = counters[bucket]; counters[bucket] += 1
                rec = synth_record(bucket, idx, base_ts=tick_ts, rng=rng)
                records.append(rec)
                # Optional within-tick duplicates (same text, different URLs)
                if dupe_rate > 0 and rng.random() < dupe_rate:
                    # create 1–2 dupes occasionally
                    for dupe_i in range(1, 1 + rng.randint(1, 2)):
                        records.append(make_dupe_of(rec, dupe_i))

        # Shuffle within the tick to avoid order bias
        rng.shuffle(records)

        # Write tick file
        path = out_dir / f"tick_{t:03}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path} with {len(records)} records")

def load_plan(path: Path) -> List[Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic validation
    if not isinstance(data, list):
        raise ValueError("plan must be a list of per-tick dicts")
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"plan[{i}] must be an object mapping bucket->count")
        # Missing buckets default to 0
        for b in BUCKETS.keys():
            row.setdefault(b, 0)
    return data

def main():
    ap = argparse.ArgumentParser(description="Generate JSON tick files for topic-branching tests")
    ap.add_argument("--out", required=True, type=Path, help="Output directory (e.g., tests/corpus)")
    ap.add_argument("--ticks", type=int, default=8, help="Number of ticks to generate")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    ap.add_argument("--dupe-rate", type=float, default=0.0,
                    help="Probability per record to emit 1–2 duplicates in the same tick (0..1)")
    ap.add_argument("--plan", type=Path, default=None,
                    help="Optional JSON plan file: list of per-tick dicts {Parent,Node,Deno,Bun,Noise}")
    args = ap.parse_args()

    if args.plan:
        plan = load_plan(args.plan)
        ticks = len(plan)
    else:
        plan = default_plan(args.ticks)
        ticks = args.ticks

    generate_ticks(args.out, ticks, plan, dupe_rate=args.dupe_rate, seed=args.seed)

if __name__ == "__main__":
    main()
