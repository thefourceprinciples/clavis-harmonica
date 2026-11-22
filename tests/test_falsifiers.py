import numpy as np
from clavis_harmonica.elements import ELEMENT_DATA
from clavis_harmonica.engine import HTEMaterialEngine
from clavis_harmonica.modes import MODES, score_modal_hypothesis
from clavis_harmonica.parser import parse_formula


def test_group_phase_beats_random_baseline():
    """Falsifier #1: Intra-family modal similarity must exceed random group-to-pitch mappings by > 3 sigma."""
    engine = HTEMaterialEngine()
    families = {
        "halides": ["NaCl", "KCl", "NaBr", "CsI"],
        "iii_v": ["GaAs", "InP", "AlN", "GaN"],
        "group_iv": ["C", "Si", "Ge"],
        "cuprates": ["YBa2Cu3O7", "La2CuO4"],
    }

    def family_coherence(vecs):
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                a, b = vecs[i].ravel(), vecs[j].ravel()
                sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        return float(np.mean(sims))

    real_scores = [
        family_coherence([engine.analyze_formula(f, "group").heatmap for f in members])
        for members in families.values()
    ]
    real_mean = float(np.mean(real_scores))

    # Null baseline: randomly assign IUPAC groups 1..18 to pitch classes 0..11
    rng = np.random.default_rng(42)
    mode_list = list(MODES.values())
    parsed_families = {
        fam: [parse_formula(f) for f in members]
        for fam, members in families.items()
    }

    null_means = []
    for _ in range(200):
        rand_grp_to_pc = {g: int(rng.integers(0, 12)) for g in range(1, 19)}
        rand_grp_to_pc = 0
        null_scores = []
        for parsed_members in parsed_families.values():
            maps = []
            for nodes in parsed_members:
                raw_h = np.zeros(12, dtype=float)
                for n in nodes:
                    raw_h[rand_grp_to_pc[n.group]] += n.weight
                h = raw_h / raw_h.sum()
                hm = np.zeros((12, len(mode_list)), dtype=float)
                for t in range(12):
                    for m_idx, m_spec in enumerate(mode_list):
                        metrics = score_modal_hypothesis(h, t, m_spec)
                        hm[t, m_idx] = metrics["match"] + 0.05 * np.tanh(metrics["fource"])
                maps.append(hm)
            null_scores.append(family_coherence(maps))
        null_means.append(float(np.mean(null_scores)))

    z_score = (real_mean - np.mean(null_means)) / np.std(null_means)
    assert z_score > 3.0, f"Falsifier triggered: Z-score {z_score:.2f} <= 3.0"


def test_stoichiometric_perturbation_stability():
    """Falsifier #2: Small doping changes must preserve basin stability without chaotic jumps."""
    engine = HTEMaterialEngine()
    modes = [
        engine.analyze_formula(f"YBa2Cu3O{x:.1f}", "group").best_mode
        for x in np.linspace(6.0, 7.0, 6)
    ]
    assert len(set(modes)) == 1, f"Unstable modal drift across doping sweep: {modes}"


def test_element_zero_substrate_boundary():
    """Falsifier #3: Element-0 acts strictly as null-reference substrate (Z=0, pi=0, omega=0)."""
    engine = HTEMaterialEngine()
    res = engine.analyze_formula("E0", "group")
    assert res.nodes.Z == 0
    assert res.nodes.pi_group == 0
    assert res.nodes.omega == 0
