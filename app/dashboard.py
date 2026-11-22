import io
import wave
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from clavis_harmonica.elements import PITCH_NAMES
from clavis_harmonica.engine import HTEMaterialEngine
from clavis_harmonica.modes import MODES

st.set_page_config(page_title="CLV-HTE-001 | Clavis Harmonica", layout="wide")
st.title("Clavis Harmonica Corporis et Materiae (`CLV-HTE-001`)")
st.caption("Tune the mode field — Exploratory Harmonic-Material Coherence & Sonification Engine")


def synthesize_chord_wav(nodes, mapping_mode: str = "group", duration: float = 2.5, sr: int = 22050) -> bytes:
    """Synthesizes an octave-layered harmonic chord WAV buffer from parsed ElementNodes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    envelope = np.minimum(1.0, t / 0.08) * np.exp(-0.9 * t)

    for n in nodes:
        pi = n.pi_group if mapping_mode == "group" else n.pi_raw
        # Map octave layer omega (0..9) around middle C (MIDI 48..84)
        midi_note = 48 + (min(n.omega, 4) * 12) + pi
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        # Fundamental + gentle 2nd harmonic weighted by stoichiometry
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


with st.sidebar:
    st.header("Field Controls")
    formula = st.text_input("Chemical Formula", value="YBa2Cu3O7")
    mapping_mode = st.radio(
        "Projection Coordinate",
        options=["group", "raw"],
        format_func=lambda x: "Periodic Group-Phase (v2)" if x == "group" else "Raw Z mod 12 (v1)",
    )
    st.markdown("---")
    st.markdown("**Presets:** `H2O`, `SiO2`, `Fe2O3`, `NaCl`, `GaAs`, `Ca10(PO4)6(OH)2`, `YBa2Cu3O7`")

engine = HTEMaterialEngine()

try:
    res = engine.analyze_formula(formula, mapping_mode=mapping_mode)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Modal Field", f"{res.best_tonic_name} {res.best_mode}")
    col2.metric("Fource Lock (F)", f"{res.fource:.3f}")
    col3.metric("Coherence (C_modal)", f"{res.c_modal:.3f}")
    col4.metric("Template Match", f"{res.match_score:.3f}")

    st.info(f"**Field Classification:** {res.field_type} — *Cognitive Signature: {res.cognitive_signature}*")

    st.subheader("Audible Lattice Chord (`Musica Instrumentalis`)")
    wav_bytes = synthesize_chord_wav(res.nodes, mapping_mode=mapping_mode)
    st.audio(wav_bytes, format="audio/wav")

    left, right = st.columns(2)

    with left:
        st.subheader("12 × 7 Modal Affinity Heatmap")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        im = ax1.imshow(res.heatmap, aspect="auto", cmap="magma")
        ax1.set_xticks(range(len(MODES)))
        ax1.set_xticklabels(list(MODES.keys()), rotation=35, ha="right")
        ax1.set_yticks(range(12))
        ax1.set_yticklabels(PITCH_NAMES)
        fig1.colorbar(im, ax=ax1, label="Selection Affinity")
        st.pyplot(fig1)

    with right:
        st.subheader("HTE Octave-Lattice Plot (π, ω)")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        pis = [n.pi_group if mapping_mode == "group" else n.pi_raw for n in res.nodes]
        omegas = [n.omega for n in res.nodes]
        sizes = [180 + n.weight * 90 for n in res.nodes]

        for i in range(len(res.nodes)):
            for j in range(i + 1, len(res.nodes)):
                w_ij = res.coupling_matrix[i, j]
                if w_ij > 0:
                    ax2.plot([pis[i], pis[j]], [omegas[i], omegas[j]], alpha=min(0.85, 0.2 + w_ij * 0.1), lw=1.5)

        ax2.scatter(pis, omegas, s=sizes, zorder=3)
        for n, x, y in zip(res.nodes, pis, omegas):
            ax2.annotate(f"{n.symbol} ({n.weight:g})", (x, y), textcoords="offset points", xytext=(6, 6))

        ax2.set_xticks(range(12))
        ax2.set_xticklabels(PITCH_NAMES, rotation=35)
        ax2.set_ylabel("Octave Layer ω = floor(Z / 12)")
        ax2.set_xlabel("Pitch Class π(Z)")
        ax2.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig2)

except Exception as exc:
    st.error(f"Parser / Engine Error: {exc}")
