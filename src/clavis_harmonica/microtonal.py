from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class EdgeStateReport:
    base_histogram: np.ndarray                      # 12-bin projected pitch-class field h
    approach_pressure: float                        # -50 to -10 cents below stable grid
    upper_pressure: float                           # +10 to +50 cents above stable grid
    blur_mass: float                                # Liminal midpoint hover (35..65 cents)
    edge_events: List[Tuple[float, str, float]]     # (cents, role, delta_cents)


def analyze_cents_field(
    cents_list: List[float],
    weights: Optional[List[float]] = None,
    deadzone_cents: float = 10.0,
    blur_band: Tuple[float, float] = (35.0, 65.0),
) -> EdgeStateReport:
    """
    Projects continuous cent measurements onto Z_12 while classifying
    7+3 microtonal edge-states (Approach, Pressure, Blur).
    """
    if weights is None:
        weights = [1.0] * len(cents_list)
    total_w = sum(weights) or 1.0
    norm_w = [w / total_w for w in weights]

    h = np.zeros(12, dtype=float)
    approach = 0.0
    pressure = 0.0
    blur = 0.0
    events: List[Tuple[float, str, float]] = []

    for c, w in zip(cents_list, norm_w):
        c_mod = c % 1200.0
        semitone_float = c_mod / 100.0
        nearest = int(round(semitone_float)) % 12
        delta = (semitone_float - round(semitone_float)) * 100.0  # [-50, +50]
        abs_d = abs(delta)

        if blur_band[0] <= abs_d <= blur_band[1]:
            # Midpoint blur splits mass across adjacent pitch classes
            low_pc = int(np.floor(semitone_float)) % 12
            high_pc = (low_pc + 1) % 12
            h[low_pc] += 0.5 * w
            h[high_pc] += 0.5 * w
            blur += w
            events.append((c, "Blur", delta))
        elif delta < -deadzone_cents:
            h[nearest] += w
            approach += w * (abs_d / 50.0)
            events.append((c, "Approach", delta))
        elif delta > deadzone_cents:
            h[nearest] += w
            pressure += w * (abs_d / 50.0)
            events.append((c, "Pressure", delta))
        else:
            h[nearest] += w
            events.append((c, "Stable", delta))

    return EdgeStateReport(
        base_histogram=h,
        approach_pressure=approach,
        upper_pressure=pressure,
        blur_mass=blur,
        edge_events=events,
    )
