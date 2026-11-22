"""
Clavis Harmonica Corporis et Materiae (CLV-HTE-001)
The Harmonic Key of Human and Material Coherence.
"""

from .elements import ELEMENT_DATA, PITCH_NAMES, ElementNode, build_element_node
from .parser import parse_formula
from .modes import MODES, ModeSpec, unit_template_vector, score_modal_hypothesis
from .microtonal import EdgeStateReport, analyze_cents_field
from .engine import HTEMaterialEngine, HTEAnalysisResult

__version__ = "0.1.0"
__all__ = [
    "ELEMENT_DATA",
    "PITCH_NAMES",
    "ElementNode",
    "build_element_node",
    "parse_formula",
    "MODES",
    "ModeSpec",
    "unit_template_vector",
    "score_modal_hypothesis",
    "EdgeStateReport",
    "analyze_cents_field",
    "HTEMaterialEngine",
    "HTEAnalysisResult",
]
