"""Negative rules for filtering documents."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class NegativeRules:
    # List of keywords/phrases that should exclude a doc
    # if they appear in its title or text.
    # Example: ["hiring", "bootcamp", "tutorial"]
    block_terms: List[str] = field(default_factory=list)

    # List of domains (hostnames) to always block for this topic.
    # Example: ["reddit.com", "quora.com"]
    block_domains: List[str] = field(default_factory=list)

    # List of doc types to ignore (dtype field on Doc).
    # Example: ["job_posting", "ad", "forum"]
    block_types: List[str] = field(default_factory=list)
