from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PromotionCheck:
    mass_ok: bool
    cohesion_ok: bool
    separation_ok: bool
    persistence_ok: bool
    # useful margins for debugging/tuning
    size: float
    m_min: float
    cohesion_ema: float
    tau_cohesion: float
    cos_parent_ema: float      # 1 - separation_ema
    tau_separation: float
    persistence: int
    persistence_min: int

    @property
    def ready(self) -> bool:
        return self.mass_ok and self.cohesion_ok and self.separation_ok and self.persistence_ok