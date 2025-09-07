"""Deduplication adapter implementations."""

from __future__ import annotations
from typing import List

from models import Doc
from protocols import Deduper


class SeenDeduper(Deduper):
    """Deduplicates documents based on seen hashes."""
    
    def drop(self, seen_hashes: set[str], docs: List[Doc]) -> List[Doc]:
        """Dedup across ticks and within the same batch."""
        out = []
        seen_in_batch = set()

        for d in docs:
            if d.hash in seen_hashes:   # already seen in past ticks
                continue
            if d.hash in seen_in_batch: # duplicate within this batch
                continue
            out.append(d)
            seen_in_batch.add(d.hash)
        return out
