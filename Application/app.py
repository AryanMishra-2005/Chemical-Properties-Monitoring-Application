"""
app.py
------
Streamlit web app for monitoring chemical thermodynamic properties.
Includes an embedded chatbot that answers natural-language questions by
calling the same tool functions used by the MCP server.

Libraries used
  - streamlit    : UI framework
  - thermo       : Thermodynamic property engine (PR EOS)
  - sqlite3      : Chemical constants database
  - init_db      : One-time DB seeding helper (ships alongside this file)
  - mcp_server   : Tool functions (get_chemical_phase, get_vapor_pressure)
"""

import re
import sqlite3
from pathlib import Path

import streamlit as st
from thermo import ChemicalConstantsPackage, PropertyCorrelationsPackage, CEOSGas, CEOSLiquid, FlashPureVLS
from thermo.eos_mix import PRMIX

from init_db import init_db, DB_PATH
from mcp_server import get_chemical_phase, get_vapor_pressure

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemical Property Monitor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ---- Global font & background ---- */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: "Segoe UI", system-ui, sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: #1B1F27;
        }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: #1B1F27;
            border-right: 1px solid #e5e7eb;
        }

        /* ---- Metric cards ---- */
        [data-testid="stMetric"] {
            background: #e1edf2;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }
        [data-testid="stMetricLabel"] { color: #57606a; font-size: 0.78rem; }
        [data-testid="stMetricValue"] { color: #1f2328; font-size: 1.55rem; font-weight: 600; }
        [data-testid="stMetricDelta"] { font-size: 0.82rem; }

        /* ---- Section headers ---- */
        .section-header {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #95d5e8;
            margin: 1.4rem 0 0.6rem;
        }

        /* ---- Phase badge ---- */
        .phase-badge {
            display: inline-block;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .phase-gas    { background: #dbeafe; color: #1d4ed8; }
        .phase-liquid { background: #dcfce7; color: #15803d; }
        .phase-mixed  { background: #fef9c3; color: #854d0e; }
        .phase-super  { background: #f3e8ff; color: #7e22ce; }

        /* ---- Chatbot ---- */
        .chat-viewport {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            max-height: 420px;
            overflow-y: auto;
            padding: 1rem;
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-bottom: 0.8rem;
        }
        .chat-bubble {
            max-width: 78%;
            padding: 0.55rem 0.95rem;
            border-radius: 16px;
            font-size: 0.88rem;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .bubble-user {
            align-self: flex-end;
            background: #1d4ed8;
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }
        .bubble-bot {
            align-self: flex-start;
            background: #ffffff;
            color: #1f2328;
            border: 1px solid #e5e7eb;
            border-bottom-left-radius: 4px;
        }
        .bubble-error {
            align-self: flex-start;
            background: #fff1f0;
            color: #c0392b;
            border: 1px solid #fca5a5;
            border-bottom-left-radius: 4px;
        }
        .chat-row-user  { display: flex; justify-content: flex-end;  }
        .chat-row-bot   { display: flex; justify-content: flex-start; }
        .chat-label {
            font-size: 0.68rem;
            color: #57606a;
            margin-bottom: 2px;
            padding: 0 0.3rem;
        }
        .chat-empty {
            text-align: center;
            color: #57606a;
            font-size: 0.84rem;
            padding: 2rem 0;
        }

        /* ---- Data table ---- */
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            background: #ffffff;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }
        .styled-table th {
            background: #f7f8fa;
            color: #57606a;
            font-weight: 600;
            padding: 0.6rem 1rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .styled-table td {
            padding: 0.6rem 1rem;
            color: #1f2328;
            border-bottom: 1px solid #f0f1f3;
        }
        .styled-table tr:last-child td { border-bottom: none; }
        .styled-table tr:hover td { background: #f7f8fa; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Database helpers ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def ensure_db() -> Path:
    """Seed the DB if it doesn't exist yet, then return its path."""
    if not DB_PATH.exists():
        init_db()
    return DB_PATH


@st.cache_data(show_spinner=False)
def load_chemicals(db_path: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM chemicals ORDER BY name").fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── Thermodynamic calculations ─────────────────────────────────────────────────

def compute_properties(chem: dict, T_K: float, P_Pa: float) -> dict:
    """
    Use thermo's PR EOS flash to compute single-component properties.

    thermo 0.6.x: ChemicalConstantsPackage.from_IDs() loads the full chemical
    database (correlation objects, etc.).  We then override Tc/Pc/omega with
    our own SQLite values so the DB is the authoritative source.
    Returns a dict of displayable results; falls back gracefully on error.
    """
    try:
        # Load full package from the chemicals library using CAS number
        consts, props = ChemicalConstantsPackage.from_IDs([chem["cas"]])

        # Override critical constants with our database values
        consts = ChemicalConstantsPackage(
            CASs=consts.CASs,
            names=consts.names,
            MWs=consts.MWs,
            Tcs=[chem["Tc"]],
            Pcs=[chem["Pc"]],
            omegas=[chem["omega"]],
            Tbs=consts.Tbs,
            Tms=consts.Tms,
        )
        # Re-fetch correlation objects (VaporPressure etc.) using overridden consts
        _, props = ChemicalConstantsPackage.from_IDs([chem["cas"]])

        eos_kwargs = dict(Tcs=[chem["Tc"]], Pcs=[chem["Pc"]], omegas=[chem["omega"]])
        gas   = CEOSGas(PRMIX,   eos_kwargs=eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
        liq   = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
        flash = FlashPureVLS(consts, props, gas=gas, liquids=[liq], solids=[])

        res   = flash.flash(T=T_K, P=P_Pa)
        phase = res.phase

        # ---- Phase label ----
        if phase == "V":
            phase_label, badge_cls = "Gas", "phase-gas"
        elif phase == "L":
            phase_label, badge_cls = "Liquid", "phase-liquid"
        elif phase in ("VL", "VLL"):
            phase_label, badge_cls = "Two-phase (VL)", "phase-mixed"
        else:
            phase_label, badge_cls = "Supercritical", "phase-super"

        bulk = res.bulk if hasattr(res, "bulk") else res

        def _safe(fn):
            try:
                v = fn()
                return v if v is not None else float("nan")
            except Exception:
                return float("nan")

        Z        = _safe(lambda: bulk.Z())
        rho_mol  = _safe(lambda: bulk.rho_mass())           # kg/m³
        H        = _safe(lambda: bulk.H()) / 1000           # kJ/mol
        S        = _safe(lambda: bulk.S())                  # J/mol·K
        Cp       = _safe(lambda: bulk.Cp()) / 1000          # kJ/mol·K
        mu       = _safe(lambda: bulk.mu())                 # Pa·s
        kappa    = _safe(lambda: bulk.k())                  # W/m·K

        return {
            "phase_label": phase_label,
            "badge_cls":   badge_cls,
            "Z":           Z,
            "rho":         rho_mol,
            "H":           H,
            "S":           S,
            "Cp":          Cp,
            "mu":          mu,
            "kappa":       kappa,
            "error":       None,
        }

    except Exception as exc:
        return {"error": str(exc)}


# Molecular weights (g/mol) — kept minimal, only what we need
_MW_TABLE = {
    "Methane": 16.043,
    "Ethane":  30.069,
    "Propane": 44.096,
    "Water":   18.015,
}

# ── Chatbot helpers ────────────────────────────────────────────────────────────

# Suggestion chips shown below the input box
_SUGGESTIONS = [
    "What phase is methane at -50 °C and 1 bar?",
    "Vapor pressure of ethanol at 78 °C?",
    "Is water a gas at 200 °C and 1 bar?",
    "Phase of propane at 20 °C, 8.5 bar?",
    "Vapor pressure of water at 100 °C",
]

# Number pattern: optional sign, digits, optional decimal
_NUM = r"[-+]?\d+(?:\.\d+)?"


# Common question prefixes to strip before name extraction
# NOTE: longer/more-specific alternatives MUST come before shorter ones
# so "what phase is" matches before "what is".
_STRIP_PREFIXES = re.compile(
    r"^\s*(?:what(?:'s)? phase (?:is|are)|what is the|what(?:'s| is| are)?|"
    r"tell me|give me|calculate|compute|find|show|"
    r"is|are|phase of|vapor pressure of|"
    r"vapour pressure of|boiling point of|saturation pressure of|"
    r"pressure of|state of)\s+",
    re.IGNORECASE,
)

# Words that are never chemical names — strip trailing matches
_TRAILING_NOISE = re.compile(
    r"\s*(?:a\s+)?(?:gas|liquid|solid|vapor|vapour|phase|state|at|"
    r"substance|compound|chemical)\s*$",
    re.IGNORECASE,
)


def _parse_and_dispatch(text: str) -> tuple[str, bool]:
    """
    Lightweight intent router.

    Returns (reply_text, is_error).
    Detects:
      - vapor / saturation / boiling pressure  → get_vapor_pressure
      - phase / state / is it a gas/liquid      → get_chemical_phase
    Extracts the first number as temperature (°C) and optional second as
    pressure (bar).  Falls back to a helpful hint when parsing fails.
    """
    t = text.lower()

    # Extract all numbers from the sentence
    nums = [float(m) for m in re.findall(_NUM, text)]

    # ---- Chemical name extraction ----
    # Step 1: strip leading question boilerplate
    cleaned = _STRIP_PREFIXES.sub("", text).strip()

    # Step 2: take everything before "at", "@", or the first digit
    name_match = re.match(
        r"([a-zA-Z][a-zA-Z0-9 \-]*?)(?:\s+at\b|\s*@|\s*[-+]?\d|$)",
        cleaned,
        re.IGNORECASE,
    )
    chemical = name_match.group(1).strip() if name_match else None

    # Step 3: strip trailing noise words ("a gas", "liquid", etc.)
    if chemical:
        chemical = _TRAILING_NOISE.sub("", chemical).strip()

    # Step 4: if still empty or multi-word mess, fall back to known-name scan
    _KNOWN = [
        "carbon dioxide", "co2", "methane", "ethane", "propane", "butane",
        "pentane", "hexane", "ethanol", "methanol", "water", "benzene",
        "toluene", "nitrogen", "oxygen", "hydrogen", "ammonia", "acetone",
    ]
    if not chemical or len(chemical.split()) > 3:
        chemical = None
        for known in _KNOWN:
            if known in t:
                chemical = known
                break

    if not chemical:
        return (
            "I couldn't identify a chemical name in your question.\n"
            "Try: \"What phase is methane at 25 °C and 2 bar?\" or "
            "\"Vapor pressure of ethanol at 20 °C\"",
            True,
        )

    if not nums:
        return (
            f"I found the chemical \"{chemical}\" but no temperature.\n"
            "Try: \"Vapor pressure of ethanol at 78 °C\"",
            True,
        )

    temp_c = nums[0]

    # Route intent
    wants_vapor = any(kw in t for kw in [
        "vapor", "vapour", "saturation", "boiling", "psat", "vp",
        "p_sat", "saturation pressure",
    ])

    if wants_vapor:
        reply = get_vapor_pressure(chemical, temp_c)
        return reply, False

    # Phase intent (default)
    press_bar = nums[1] if len(nums) >= 2 else 1.01325
    reply = get_chemical_phase(chemical, temp_c, press_bar)
    return reply, False


def _render_chat(messages: list[dict]) -> None:
    """Render the full conversation history as styled bubbles."""
    if not messages:
        st.markdown(
            '<div class="chat-viewport">'
            '<p class="chat-empty">No messages yet — ask a question below.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    bubbles = ""
    for msg in messages:
        role  = msg["role"]        # "user" | "bot" | "error"
        body  = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        if role == "user":
            bubbles += (
                f'<div class="chat-row-user">'
                f'<div class="chat-bubble bubble-user">{body}</div>'
                f'</div>'
            )
        elif role == "error":
            bubbles += (
                f'<div class="chat-row-bot">'
                f'<div class="chat-bubble bubble-error">⚠ {body}</div>'
                f'</div>'
            )
        else:
            bubbles += (
                f'<div class="chat-row-bot">'
                f'<div class="chat-bubble bubble-bot">{body}</div>'
                f'</div>'
            )

    st.markdown(
        f'<div class="chat-viewport">{bubbles}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

db_path = ensure_db()
chemicals = load_chemicals(str(db_path))
chem_names = [c["name"] for c in chemicals]
chem_map   = {c["name"]: c for c in chemicals}

with st.sidebar:
    st.markdown("## ⚗️ Chemical Monitor")
    st.markdown("---")

    # Chemical selector
    st.markdown('<p class="section-header">Compound</p>', unsafe_allow_html=True)
    selected_name = st.selectbox(
        label="Select a chemical",
        options=chem_names,
        index=0,
        label_visibility="collapsed",
    )
    chem = chem_map[selected_name]

    # Temperature slider
    st.markdown('<p class="section-header">Temperature</p>', unsafe_allow_html=True)
    T_C = st.slider(
        "Temperature (°C)",
        min_value=-100,
        max_value=500,
        value=25,
        step=1,
        format="%d °C",
        label_visibility="visible",
    )
    T_K = T_C + 273.15

    # Pressure input
    st.markdown('<p class="section-header">Pressure</p>', unsafe_allow_html=True)
    P_bar = st.number_input(
        "Pressure (bar)",
        min_value=0.01,
        max_value=500.0,
        value=1.0133,
        step=0.5,
        format="%.4f",
    )
    P_Pa = P_bar * 1e5

    st.markdown("---")
    st.caption("Data: NIST / Perry's · EOS: Peng–Robinson")


# ── Session state ──────────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat = st.tabs(["⚗️  Dashboard", "💬  Chatbot"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:

    st.markdown(f"## {chem['formula']} — {selected_name}")
    st.markdown(
        f"**T** = {T_C} °C &nbsp;({T_K:.2f} K) &emsp; "
        f"**P** = {P_bar:.4f} bar &nbsp;({P_Pa/1e6:.4f} MPa)"
    )

    # ── Critical constants ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">Critical Constants (from database)</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Critical Temperature  Tₓ", f"{chem['Tc']:.2f} K",
                  delta=f"ΔT = {T_K - chem['Tc']:+.1f} K from Tₓ")
    with col2:
        st.metric("Critical Pressure  Pₓ", f"{chem['Pc']/1e6:.4f} MPa",
                  delta=f"ΔP = {P_Pa/1e6 - chem['Pc']/1e6:+.4f} MPa from Pₓ")
    with col3:
        st.metric("Acentric Factor  ω", f"{chem['omega']:.5f}")

    # ── Flash calculation ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">Thermodynamic Properties (PR EOS Flash)</p>', unsafe_allow_html=True)

    with st.spinner("Computing thermodynamic properties…"):
        props = compute_properties(chem, T_K, P_Pa)

    if props["error"]:
        st.error(f"Flash calculation failed: {props['error']}")
    else:
        st.markdown(
            f'<span class="phase-badge {props["badge_cls"]}">Phase: {props["phase_label"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Compressibility Z", f"{props['Z']:.5f}")
        with c2:
            st.metric("Density ρ", f"{props['rho']:.3f} kg/m³")
        with c3:
            st.metric("Enthalpy H", f"{props['H']:.4f} kJ/mol")
        with c4:
            st.metric("Entropy S", f"{props['S']:.4f} J/mol·K")

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Heat Capacity Cₚ", f"{props['Cp']:.5f} kJ/mol·K")
        with c6:
            mu_val = props["mu"]
            st.metric("Viscosity μ", f"{mu_val:.4e} Pa·s" if mu_val == mu_val else "N/A")
        with c7:
            k_val = props["kappa"]
            st.metric("Thermal Cond. κ", f"{k_val:.4f} W/m·K" if k_val == k_val else "N/A")

    # ── Reference table ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">All Stored Compounds</p>', unsafe_allow_html=True)

    rows_html = ""
    for c in chemicals:
        highlight = ' style="background:#f0f6ff;"' if c["name"] == selected_name else ""
        rows_html += (
            f"<tr{highlight}>"
            f"<td>{c['name']}</td>"
            f"<td><code>{c['formula']}</code></td>"
            f"<td>{c['cas']}</td>"
            f"<td>{c['Tc']:.2f}</td>"
            f"<td>{c['Pc']/1e6:.4f}</td>"
            f"<td>{c['omega']:.5f}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="styled-table">
          <thead>
            <tr>
              <th>Name</th><th>Formula</th><th>CAS</th>
              <th>Tₓ (K)</th><th>Pₓ (MPa)</th><th>ω</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.caption(
        "Built with [Streamlit](https://streamlit.io) · "
        "[thermo](https://github.com/CalebBell/thermo) · "
        "Peng–Robinson EOS"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chatbot
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    st.markdown("### Chemical Properties Chatbot")
    st.markdown(
        "Ask a question in plain English. The chatbot calls the same "
        "**PR EOS** tools as the MCP server to answer you."
    )

    # ── Conversation history ────────────────────────────────────────────────
    _render_chat(st.session_state.chat_messages)

    # ── Suggestion chips ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Try one of these</p>', unsafe_allow_html=True)
    chip_cols = st.columns(len(_SUGGESTIONS))
    for idx, (col, suggestion) in enumerate(zip(chip_cols, _SUGGESTIONS)):
        with col:
            if st.button(
                suggestion,
                key=f"chip_{idx}",
                use_container_width=True,
            ):
                # Treat the chip click exactly like a typed submission
                st.session_state.chat_messages.append(
                    {"role": "user", "content": suggestion}
                )
                reply, is_err = _parse_and_dispatch(suggestion)
                st.session_state.chat_messages.append(
                    {"role": "error" if is_err else "bot", "content": reply}
                )
                st.session_state.chat_input_key += 1
                st.rerun()

    # ── Text input ──────────────────────────────────────────────────────────
    with st.form(
        key=f"chat_form_{st.session_state.chat_input_key}",
        clear_on_submit=True,
    ):
        user_input = st.text_input(
            "Your question",
            placeholder='e.g. "What phase is propane at -10 °C and 5 bar?"',
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_input.strip()}
        )
        reply, is_err = _parse_and_dispatch(user_input.strip())
        st.session_state.chat_messages.append(
            {"role": "error" if is_err else "bot", "content": reply}
        )
        st.session_state.chat_input_key += 1
        st.rerun()

    # ── Clear button ────────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_messages = []
            st.session_state.chat_input_key += 1
            st.rerun()
