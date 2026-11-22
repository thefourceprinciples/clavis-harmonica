import io
import json
import wave
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from clavis_harmonica.elements import PITCH_NAMES
from clavis_harmonica.engine import HTEMaterialEngine
from clavis_harmonica.microtonal import analyze_cents_field
from clavis_harmonica.modes import MODES, score_modal_hypothesis

st.set_page_config(page_title="CLV-HTE-001 | Interactive HTE Engine", layout="wide")
st.title("Clavis Harmonica Corporis et Materiae (`CLV-HTE-001`)")
st.caption("Interactive HTE Coherence Engine — Modal Topology, Material Comparison & 7+3 Microtonal Lab")


def synthesize_chord_wav(nodes, mapping_mode: str = "group", duration: float = 2.2, sr: int = 22050) -> bytes:
    """Synthesizes an octave-layered harmonic chord WAV buffer from parsed ElementNodes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    envelope = np.minimum(1.0, t / 0.08) * np.exp(-0.9 * t)

    for n in nodes:
        pi = n.pi_group if mapping_mode == "group" else n.pi_raw
        midi_note = 48 + (min(n.omega, 4) * 12) + pi
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        wave_i = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
        signal += n.weight * wave_i

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = (signal / peak) * envelope * 0.85

    pcm = np.int16(signal * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def synthesize_cents_wav(cents_list, weights, duration: float = 2.2, sr: int = 22050) -> bytes:
    """Synthesizes a microtonal cent-scale chord relative to C4 (261.63 Hz)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    envelope = np.minimum(1.0, t / 0.08) * np.exp(-0.9 * t)

    for c, w in zip(cents_list, weights):
        freq = 261.6256 * (2.0 ** (c / 1200.0))
        signal += w * (np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(4 * np.pi * freq * t))

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = (signal / peak) * envelope * 0.85

    pcm = np.int16(signal * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def plot_heatmap(heatmap: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(heatmap, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(list(MODES.keys()), rotation=35, ha="right")
    ax.set_yticks(range(12))
    ax.set_yticklabels(PITCH_NAMES)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Selection Affinity")
    return fig


def plot_lattice(res, mapping_mode: str):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    pis = [n.pi_group if mapping_mode == "group" else n.pi_raw for n in res.nodes]
    omegas = [n.omega for n in res.nodes]
    sizes = [180 + n.weight * 90 for n in res.nodes]

    for i in range(len(res.nodes)):
        for j in range(i + 1, len(res.nodes)):
            w_ij = res.coupling_matrix[i, j]
            if w_ij > 0:
                ax.plot([pis[i], pis[j]], [omegas[i], omegas[j]], alpha=min(0.85, 0.2 + w_ij * 0.1), lw=1.5)

    ax.scatter(pis, omegas, s=sizes, zorder=3)
    for n, x, y in zip(res.nodes, pis, omegas):
        ax.annotate(f"{n.symbol} ({n.weight:g})", (x, y), textcoords="offset points", xytext=(6, 6))

    ax.set_xticks(range(12))
    ax.set_xticklabels(PITCH_NAMES, rotation=35)
    ax.set_ylabel("Octave Layer ω = floor(Z / 12)")
    ax.set_xlabel("Pitch Class π(Z)")
    ax.set_title(f"Octave-Lattice Coupling :: {res.label}")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


with st.sidebar:
    st.header("Engine Parameters")
    mapping_mode = st.radio(
        "Projection Coordinate",
        options=["group", "raw"],
        format_func=lambda x: "Periodic Group-Phase (v2)" if x == "group" else "Raw Z mod 12 (v1)",
    )
    alpha = st.slider("Pitch-Class Distance Weight (α)", 0.2, 2.5, 1.0, 0.1)
    beta = st.slider("Octave Separation Weight (β)", 0.1, 2.0, 0.5, 0.1)
    st.markdown("---")
    st.markdown("**Quick Presets:**\n- Ionic: `NaCl`, `KCl`, `CaF2`\n- Semiconductor: `GaAs`, `InP`, `SiO2`\n- Cuprate: `YBa2Cu3O7`, `La2CuO4`\n- Substrate: `E0`")

engine = HTEMaterialEngine(alpha=alpha, beta=beta)

tab1, tab2, tab3 = st.tabs([
    "1. Material Field Observatory",
    "2. A/B Material Comparator",
    "3. 7+3 Microtonal Edge Lab",
])

# =====================================================================
# TAB 1: SINGLE-MATERIAL OBSERVATORY + LEDGER EXPORT
# =====================================================================
with tab1:
    formula = st.text_input("Enter Chemical Formula", value="YBa2Cu3O7", key="obs_formula")
    try:
        res = engine.analyze_formula(formula, mapping_mode=mapping_mode)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Modal Field", f"{res.best_tonic_name} {res.best_mode}")
        c2.metric("Fource Lock (F)", f"{res.fource:.3f}")
        c3.metric("Coherence (C_modal)", f"{res.c_modal:.3f}")
        c4.metric("Template Match", f"{res.match_score:.3f}")

        st.info(f"**Field Type:** {res.field_type} | **Cognitive Signature:** *{res.cognitive_signature}*")

        audio_col, export_col = st.columns([2, 1])
        with audio_col:
            st.audio(synthesize_chord_wav(res.nodes, mapping_mode=mapping_mode), format="audio/wav")
        with export_col:
            ledger_payload = {
                "schema": "CLV-HTE-001-LEDGER-v0.1",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "formula": res.label,
                "mapping_mode": res.mapping_mode,
                "optimal_tonic": res.best_tonic_name,
                "optimal_mode": res.best_mode,
                "field_type": res.field_type,
                "fource_lock": round(res.fource, 6),
                "c_modal": round(res.c_modal, 6),
                "match_score": round(res.match_score, 6),
                "claim_boundary": "Exploratory HTE harmonic-material overlay candidate; not a chemical proof.",
                "nodes": [
                    {
                        "symbol": n.symbol,
                        "Z": n.Z,
                        "group": n.group,
                        "weight": n.weight,
                        "pi": n.pi_group if mapping_mode == "group" else n.pi_raw,
                        "omega": n.omega,
                    }
                    for n in res.nodes
                ],
            }
            st.download_button(
                label="Download JSON Experiment Ledger",
                data=json.dumps(ledger_payload, indent=2),
                file_name=f"{res.label}_hte_ledger.json",
                mime="application/json",
            )

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_heatmap(res.heatmap, f"12 × 7 Modal Affinity :: {res.label}"))
        with right:
            st.pyplot(plot_lattice(res, mapping_mode))

    except Exception as exc:
        st.error(f"Engine Error: {exc}")

# =====================================================================
# TAB 2: SIDE-BY-SIDE MATERIAL COMPARATOR (A/B DOPING & FAMILY DIFF)
# =====================================================================
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        formula_a = st.text_input("Material A (Reference)", value="YBa2Cu3O6.0", key="cmp_a")
    with col_b:
        formula_b = st.text_input("Material B (Target / Doped)", value="YBa2Cu3O7.0", key="cmp_b")

    try:
        res_a = engine.analyze_formula(formula_a, mapping_mode=mapping_mode)
        res_b = engine.analyze_formula(formula_b, mapping_mode=mapping_mode)

        vec_a, vec_b = res_a.heatmap.ravel(), res_b.heatmap.ravel()
        hm_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
        delta_f = res_b.fource - res_a.fource
        delta_c = res_b.c_modal - res_a.c_modal

        m1, m2, m3 = st.columns(3)
        m1.metric("Modal Field Shift", f"{res_a.best_tonic_name} {res_a.best_mode} → {res_b.best_tonic_name} {res_b.best_mode}")
        m2.metric("Heatmap Cosine Similarity", f"{hm_sim:.4f}")
        m3.metric("Fource Shift (ΔF)", f"{delta_f:+.4f}", delta=f"ΔC_modal: {delta_c:+.4f}")

        aud_a, aud_b = st.columns(2)
        with aud_a:
            st.markdown(f"**Chord A:** `{res_a.label}` ({res_a.best_tonic_name} {res_a.best_mode}, F={res_a.fource:.3f})")
            st.audio(synthesize_chord_wav(res_a.nodes, mapping_mode=mapping_mode), format="audio/wav")
            st.pyplot(plot_heatmap(res_a.heatmap, f"Heatmap A :: {res_a.label}"))
        with aud_b:
            st.markdown(f"**Chord B:** `{res_b.label}` ({res_b.best_tonic_name} {res_b.best_mode}, F={res_b.fource:.3f})")
            st.audio(synthesize_chord_wav(res_b.nodes, mapping_mode=mapping_mode), format="audio/wav")
            st.pyplot(plot_heatmap(res_b.heatmap, f"Heatmap B :: {res_b.label}"))

    except Exception as exc:
        st.error(f"Comparison Error: {exc}")

# =====================================================================
# TAB 3: 7 + 3 MICROTONAL EDGE-STATE & CENT SYNTHESIZER
# =====================================================================
with tab3:
    st.markdown("Bend a 4-tone modal anchor (`Root`, `3rd`, `5th`, `7th`) in cents to observe **Approach**, **Pressure**, and **Blur** edge-states.")
    s1, s2, s3, s4 = st.columns(4)
    c_root = s1.slider("Tone 1 (Root, ¢)", -50, 50, 0)
    c_third = s2.slider("Tone 2 (3rd @ 400¢ + Δ)", -150, 50, 0)
    c_fifth = s3.slider("Tone 3 (5th @ 700¢ + Δ)", -150, 50, 0)
