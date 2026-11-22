from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class ModeSpec:
    name: str
    intervals: Tuple[int, ...]
    characteristic_degree: int
    third_degree: int
    fifth_degree: int
    field_type: str
    cognitive_signature: str


MODES: Dict[str, ModeSpec] = {
    "Ionian": ModeSpec(
        name="Ionian",
        intervals=(0, 2, 4, 5, 7, 9, 11),
        characteristic_degree=11,
        third_degree=4,
        fifth_degree=7,
        field_type="Structural Clarity Field",
        cognitive_signature="stable prediction, closure",
    ),
    "Dorian": ModeSpec(
        name="Dorian",
        intervals=(0, 2, 3, 5, 7, 9, 10),
        characteristic_degree=9,
        third_degree=3,
        fifth_degree=7,
        field_type="Restoration Field",
        cognitive_signature="recovery under load",
    ),
    "Phrygian": ModeSpec(
        name="Phrygian",
        intervals=(0, 1, 3, 5, 7, 8, 10),
        characteristic_degree=1,
        third_degree=3,
        fifth_degree=7,
        field_type="Boundary / Threshold Field",
        cognitive_signature="boundary alertness",
    ),
    "Lydian": ModeSpec(
        name="Lydian",
        intervals=(0, 2, 4, 6, 7, 9, 11),
        characteristic_degree=6,
        third_degree=4,
        fifth_degree=7,
        field_type="Expansion / Levitation Field",
        cognitive_signature="upward spatiality",
    ),
    "Mixolydian": ModeSpec(
        name="Mixolydian",
        intervals=(0, 2, 4, 5, 7, 9, 10),
        characteristic_degree=10,
        third_degree=4,
        fifth_degree=7,
        field_type="Continuity Field",
        cognitive_signature="social continuity",
    ),
    "Aeolian": ModeSpec(
        name="Aeolian",
        intervals=(0, 2, 3, 5, 7, 8, 10),
        characteristic_degree=8,
        third_degree=3,
        fifth_degree=7,
        field_type="Memory / Gravitational Field",
        cognitive_signature="reflective descent",
    ),
    "Locrian": ModeSpec(
        name="Locrian",
        intervals=(0, 1, 3, 5, 6, 8, 10),
        characteristic_degree=6,
        third_degree=3,
        fifth_degree=6,
        field_type="Fracture / Collapse Field",
        cognitive_signature="center-loss",
    ),
}


def unit_template_vector(tonic: int, mode: ModeSpec) -> np.ndarray:
    """Constructs an L2-normalized template vector to prevent single-note norm bias."""
    vec = np.zeros(12, dtype=float)
    for deg in mode.intervals:
        vec[(tonic + deg) % 12] = 1.0
    vec[tonic] = 2.0
    vec[(tonic + mode.fifth_degree) % 12] += 0.5
    vec[(tonic + mode.characteristic_degree) % 12] += 0.6
    vec[(tonic + mode.third_degree) % 12] += 0.3
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def score_modal_hypothesis(
    h: np.ndarray,
    tonic: int,
    mode: ModeSpec,
    epsilon: float = 0.05,
    H_consistency: float = 1.0,
) -> Dict[str, float]:
    """Evaluates Template Match, C_modal = O * E * H * X, and Fource F."""
    template_pcs = {(tonic + deg) % 12 for deg in mode.intervals}
    t_vec = unit_template_vector(tonic, mode)

    norm_h = np.linalg.norm(h)
    cos_sim = float(np.dot(h, t_vec) / norm_h) if norm_h > 0 else 0.0
    off_mass = float(sum(h[k] for k in range(12) if k not in template_pcs))
    match_score = max(0.0, cos_sim - 0.5 * off_mass)

    h_root = float(h[tonic])
    h_fifth = float(h[(tonic + mode.fifth_degree) % 12])
    h_third = float(h[(tonic + mode.third_degree) % 12])
    h_char = float(h[(tonic + mode.characteristic_degree) % 12])

    O = min(1.0, (0.50 * h_root + 0.30 * h_fifth + 0.20 * h_third) * 2.0)
    E = 1.0 - off_mass
    H = H_consistency
    X = min(1.0, 0.25 + 1.5 * h_char)
    c_modal = O * E * H * X

    m2_pressure = 0.0
    tri_pressure = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            interval = (j - i) % 12
            pair_mass = float(h[i] * h[j])
            if interval in (1, 11):
                m2_pressure += pair_mass
            elif interval == 6:
                tri_pressure += pair_mass

    numerator = h_root * (h_char + 0.15 * h_fifth)
    denominator = epsilon + off_mass + m2_pressure + tri_pressure
    fource = numerator / denominator

    return {
        "match": match_score,
        "c_modal": c_modal,
        "O": O,
        "E": E,
        "H": H,
        "X": X,
        "fource": fource,
        "off": off_mass,
        "m2": m2_pressure,
        "tri": tri_pressure,
    }
