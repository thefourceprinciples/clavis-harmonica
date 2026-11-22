import math
from dataclasses import dataclass
from typing import Dict, List, Literal
import numpy as np
from .elements import ElementNode, PITCH_NAMES
from .modes import MODES, score_modal_hypothesis
from .parser import parse_formula

MappingMode = Literal["group", "raw"]


@dataclass
class HTEAnalysisResult:
    label: str
    mapping_mode: MappingMode
    nodes: List[ElementNode]
    histogram: np.ndarray
    best_tonic: int
    best_tonic_name: str
    best_mode: str
    match_score: float
    c_modal: float
    fource: float
    field_type: str
    cognitive_signature: str
    heatmap: np.ndarray
    coupling_matrix: np.ndarray


class HTEMaterialEngine:
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, epsilon: float = 0.05):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.mode_names = list(MODES.keys())

    def _get_pi(self, node: ElementNode, mapping_mode: MappingMode) -> int:
        return node.pi_group if mapping_mode == "group" else node.pi_raw

    def pairwise_coupling(self, nodes: List[ElementNode], mapping_mode: MappingMode = "group") -> np.ndarray:
        n = len(nodes)
        W = np.zeros((n, n), dtype=float)
        consonance = {0: 1.5, 7: 1.4, 5: 1.3, 4: 1.15, 3: 1.15, 9: 1.1, 8: 1.1}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pi_i = self._get_pi(nodes[i], mapping_mode)
                pi_j = self._get_pi(nodes[j], mapping_mode)
                d_pc = min(abs(pi_i - pi_j), 12 - abs(pi_i - pi_j))
                d_oct = abs(nodes[i].omega - nodes[j].omega)
                dist = self.alpha * d_pc + self.beta * d_oct
                interval = (pi_j - pi_i) % 12
                W[i, j] = (nodes[i].weight * nodes[j].weight * consonance.get(interval, 0.85)) / (1.0 + dist)
        return W

    def analyze_formula(self, formula: str, mapping_mode: MappingMode = "group") -> HTEAnalysisResult:
        nodes = parse_formula(formula)
        raw_h = np.zeros(12, dtype=float)
        for n in nodes:
            raw_h[self._get_pi(n, mapping_mode)] += n.weight
        h = raw_h / raw_h.sum() if raw_h.sum() > 0 else raw_h

        heatmap = np.zeros((12, len(self.mode_names)), dtype=float)
        best_score = -1.0
        best_T, best_M = 0, "Ionian"
        best_metrics: Dict[str, float] = {}

        for t in range(12):
            for m_idx, m_name in enumerate(self.mode_names):
                metrics = score_modal_hypothesis(h, t, MODES[m_name], epsilon=self.epsilon)
                score = metrics["match"] + 0.05 * math.tanh(metrics["fource"])
                heatmap[t, m_idx] = score
                if score > best_score:
                    best_score = score
                    best_T, best_M = t, m_name
                    best_metrics = metrics

        spec = MODES[best_M]
        return HTEAnalysisResult(
            label=formula,
            mapping_mode=mapping_mode,
            nodes=nodes,
            histogram=h,
            best_tonic=best_T,
            best_tonic_name=PITCH_NAMES[best_T],
            best_mode=best_M,
            match_score=best_metrics["match"],
            c_modal=best_metrics["c_modal"],
            fource=best_metrics["fource"],
            field_type=spec.field_type,
            cognitive_signature=spec.cognitive_signature,
            heatmap=heatmap,
            coupling_matrix=self.pairwise_coupling(nodes, mapping_mode),
        )
