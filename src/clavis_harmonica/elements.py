from dataclasses import dataclass
from typing import Dict, Tuple

PITCH_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]

# (Z, IUPAC Group 1..18) — Group 0 reserved for Element-0 substrate; f-block mapped to Group 3
ELEMENT_DATA: Dict[str, Tuple[int, int]] = {
    "E0": (0, 0),   # Element-0: Formal coherence-substrate symbol (non-physical)
    "H": (1, 1), "He": (2, 18),
    "Li": (3, 1), "Be": (4, 2), "B": (5, 13), "C": (6, 14), "N": (7, 15), "O": (8, 16), "F": (9, 17), "Ne": (10, 18),
    "Na": (11, 1), "Mg": (12, 2), "Al": (13, 13), "Si": (14, 14), "P": (15, 15), "S": (16, 16), "Cl": (17, 17), "Ar": (18, 18),
    "K": (19, 1), "Ca": (20, 2), "Sc": (21, 3), "Ti": (22, 4), "V": (23, 5), "Cr": (24, 6), "Mn": (25, 7),
    "Fe": (26, 8), "Co": (27, 9), "Ni": (28, 10), "Cu": (29, 11), "Zn": (30, 12), "Ga": (31, 13), "Ge": (32, 14),
    "As": (33, 15), "Se": (34, 16), "Br": (35, 17), "Kr": (36, 18),
    "Rb": (37, 1), "Sr": (38, 2), "Y": (39, 3), "Zr": (40, 4), "Nb": (41, 5), "Mo": (42, 6), "Tc": (43, 7),
    "Ru": (44, 8), "Rh": (45, 9), "Pd": (46, 10), "Ag": (47, 11), "Cd": (48, 12), "In": (49, 13), "Sn": (50, 14),
    "Sb": (51, 15), "Te": (52, 16), "I": (53, 17), "Xe": (54, 18),
    "Cs": (55, 1), "Ba": (56, 2),
    "La": (57, 3), "Ce": (58, 3), "Pr": (59, 3), "Nd": (60, 3), "Pm": (61, 3), "Sm": (62, 3), "Eu": (63, 3),
    "Gd": (64, 3), "Tb": (65, 3), "Dy": (66, 3), "Ho": (67, 3), "Er": (68, 3), "Tm": (69, 3), "Yb": (70, 3), "Lu": (71, 3),
    "Hf": (72, 4), "Ta": (73, 5), "W": (74, 6), "Re": (75, 7), "Os": (76, 8), "Ir": (77, 9), "Pt": (78, 10),
    "Au": (79, 11), "Hg": (80, 12), "Tl": (81, 13), "Pb": (82, 14), "Bi": (83, 15), "Po": (84, 16), "At": (85, 17), "Rn": (86, 18),
    "Fr": (87, 1), "Ra": (88, 2),
    "Ac": (89, 3), "Th": (90, 3), "Pa": (91, 3), "U": (92, 3), "Np": (93, 3), "Pu": (94, 3), "Am": (95, 3),
    "Cm": (96, 3), "Bk": (97, 3), "Cf": (98, 3), "Es": (99, 3), "Fm": (100, 3), "Md": (101, 3), "No": (102, 3), "Lr": (103, 3),
    "Rf": (104, 4), "Db": (105, 5), "Sg": (106, 6), "Bh": (107, 7), "Hs": (108, 8), "Mt": (109, 9), "Ds": (110, 10),
    "Rg": (111, 11), "Cn": (112, 12), "Nh": (113, 13), "Fl": (114, 14), "Mc": (115, 15), "Lv": (116, 16), "Ts": (117, 17), "Og": (118, 18),
}


@dataclass(frozen=True)
class ElementNode:
    symbol: str
    Z: int
    group: int
    weight: float
    pi_raw: int     # Z mod 12
    pi_group: int   # Periodic group-phase projection onto Z_12
    omega: int      # Octave / shell layer index floor(Z / 12)


def build_element_node(symbol: str, weight: float = 1.0) -> ElementNode:
    if symbol not in ELEMENT_DATA:
        raise ValueError(f"Unknown element symbol: '{symbol}'")
    z, grp = ELEMENT_DATA[symbol]
    pi_raw = z % 12
    pi_group = 0 if grp == 0 else int(round(((grp - 1) * 12) / 18.0)) % 12
    omega = z // 12
    return ElementNode(
        symbol=symbol,
        Z=z,
        group=grp,
        weight=weight,
        pi_raw=pi_raw,
        pi_group=pi_group,
        omega=omega,
    )
