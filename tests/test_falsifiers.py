import numpy as np
from clavis_harmonica.engine import HTEMaterialEngine


def test_group_phase_beats_random_baseline():
    """Falsifier #1: Intra-family modal similarity must exceed random permutations by > 3 sigma."""
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

    rng = np.random.default_rng(42)
    null_means = []
    for _ in range(300):
        perm = rng.permutation(12)
        null_scores = []
        for members in families.values():
            shuffled_maps = []
            for f in members:
                hm = engine.analyze_formula(f, "group").heatmap
                shuffled_maps.append(hm[perm, :])
            null_scores.append(family_coherence(shuffled_maps))
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
    assert res.nodes[0].Z == 0
    assert res.nodes[0].pi_group == 0
    assert res.nodes[0].omega == 0
