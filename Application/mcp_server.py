"""
mcp_server.py
-------------
Python MCP server exposing chemical-property tools via the MCP 2.x protocol.
Compatible with watsonx Orchestrate (stdio transport).

Tools
  • get_chemical_phase(name, temp_c, press_bar)
      Returns the thermodynamic phase (Gas / Liquid / Two-phase / Supercritical)
      of any pure compound at the given temperature and pressure, computed with
      the Peng–Robinson equation of state.

  • get_vapor_pressure(name, temp_c)
      Returns the saturation (vapour) pressure of a pure compound at the given
      temperature using the correlation stored in the thermo / chemicals database.

Runtime dependencies (already in requirements.txt)
  thermo>=0.2.27, fluids>=1.0.26
MCP SDK
  mcp>=2.0  (installed separately: pip install mcp)
"""

import sys
from chemicals import CAS_from_any
from thermo import (
    ChemicalConstantsPackage,
    CEOSGas,
    CEOSLiquid,
    FlashPureVLS,
)
from thermo.eos_mix import PRMIX
from mcp.server.mcpserver import MCPServer

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
server = MCPServer(
    name="chemical-properties",
    title="Chemical Property Tools",
    description=(
        "Thermodynamic property tools powered by the thermo / Peng–Robinson "
        "EOS. Accepts any IUPAC name, common name, or CAS number."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PHASE_LABELS: dict[str, str] = {
    "V":   "Gas",
    "L":   "Liquid",
    "VL":  "Two-phase (vapour + liquid)",
    "VLL": "Two-phase (vapour + two liquids)",
    "S":   "Solid",
}


def _build_flash(cas: str) -> FlashPureVLS:
    """Build a PR-EOS FlashPureVLS object for a single compound."""
    consts, props = ChemicalConstantsPackage.from_IDs([cas])
    eos_kw = dict(Tcs=consts.Tcs, Pcs=consts.Pcs, omegas=consts.omegas)
    gas = CEOSGas(
        PRMIX,
        eos_kwargs=eos_kw,
        HeatCapacityGases=props.HeatCapacityGases,
    )
    liq = CEOSLiquid(
        PRMIX,
        eos_kwargs=eos_kw,
        HeatCapacityGases=props.HeatCapacityGases,
    )
    return FlashPureVLS(consts, props, gas=gas, liquids=[liq], solids=[]), props


def _resolve_cas(name: str) -> str:
    """Resolve a chemical name / CAS string to a canonical CAS number."""
    return CAS_from_any(name.strip())


# ---------------------------------------------------------------------------
# Tool: get_chemical_phase
# ---------------------------------------------------------------------------

@server.tool(
    name="get_chemical_phase",
    description=(
        "Return the thermodynamic phase of a pure chemical compound at the "
        "specified temperature and pressure using the Peng–Robinson EOS.\n\n"
        "Parameters\n"
        "----------\n"
        "name      : Chemical name or CAS number (e.g. 'water', 'methane', '74-82-8').\n"
        "temp_c    : Temperature in degrees Celsius.\n"
        "press_bar : Absolute pressure in bar (1 bar ≈ 0.987 atm).\n\n"
        "Returns\n"
        "-------\n"
        "A plain-text description of the phase, key conditions, and "
        "compressibility factor Z."
    ),
)
def get_chemical_phase(name: str, temp_c: float, press_bar: float) -> str:
    """
    Compute and return the thermodynamic phase of *name* at *temp_c* °C
    and *press_bar* bar.
    """
    try:
        cas   = _resolve_cas(name)
        T_K   = temp_c + 273.15
        P_Pa  = press_bar * 1e5

        if T_K <= 0:
            return f"Error: Temperature {temp_c} °C converts to {T_K:.2f} K, which is non-physical."
        if P_Pa <= 0:
            return f"Error: Pressure must be positive (got {press_bar} bar)."

        flash, _ = _build_flash(cas)
        result   = flash.flash(T=T_K, P=P_Pa)
        bulk     = result.bulk if hasattr(result, "bulk") else result

        raw_phase  = result.phase
        phase_name = _PHASE_LABELS.get(raw_phase, f"Unknown ({raw_phase})")

        try:
            Z_str = f"{bulk.Z():.5f}"
        except Exception:
            Z_str = "N/A"

        return (
            f"Chemical  : {name}  (CAS {cas})\n"
            f"Conditions: T = {temp_c:.2f} °C ({T_K:.2f} K), "
            f"P = {press_bar:.4f} bar ({P_Pa / 1e6:.6f} MPa)\n"
            f"Phase     : {phase_name}\n"
            f"Z factor  : {Z_str}"
        )

    except Exception as exc:
        return f"Error computing phase for '{name}': {exc}"


# ---------------------------------------------------------------------------
# Tool: get_vapor_pressure
# ---------------------------------------------------------------------------

@server.tool(
    name="get_vapor_pressure",
    description=(
        "Return the vapour (saturation) pressure of a pure chemical compound "
        "at a given temperature using the correlation in the thermo/chemicals "
        "database (Antoine / Wagner / extended Antoine as appropriate).\n\n"
        "Parameters\n"
        "----------\n"
        "name   : Chemical name or CAS number (e.g. 'ethanol', 'propane', '74-98-6').\n"
        "temp_c : Temperature in degrees Celsius.\n\n"
        "Returns\n"
        "-------\n"
        "Vapour pressure in Pa, kPa, and bar, plus the temperature in K."
    ),
)
def get_vapor_pressure(name: str, temp_c: float) -> str:
    """
    Return the saturation pressure of *name* at *temp_c* °C.
    """
    try:
        cas  = _resolve_cas(name)
        T_K  = temp_c + 273.15

        if T_K <= 0:
            return f"Error: Temperature {temp_c} °C converts to {T_K:.2f} K, which is non-physical."

        _, props = ChemicalConstantsPackage.from_IDs([cas])
        vp_obj   = props.VaporPressures[0]
        psat_Pa  = vp_obj.T_dependent_property(T_K)

        if psat_Pa is None:
            return (
                f"Vapour pressure for '{name}' at {temp_c:.2f} °C "
                f"could not be evaluated (outside correlation range)."
            )

        return (
            f"Chemical        : {name}  (CAS {cas})\n"
            f"Temperature     : {temp_c:.2f} °C  ({T_K:.2f} K)\n"
            f"Vapour pressure : {psat_Pa:.4f} Pa\n"
            f"                  {psat_Pa / 1e3:.6f} kPa\n"
            f"                  {psat_Pa / 1e5:.8f} bar"
        )

    except Exception as exc:
        return f"Error computing vapour pressure for '{name}': {exc}"


# ---------------------------------------------------------------------------
# Entry point — stdio transport (watsonx Orchestrate compatible)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # All diagnostic output MUST go to stderr; stdout is the MCP protocol channel.
    print("chemical-properties MCP server starting (stdio transport)…", file=sys.stderr)
    server.run(transport="stdio")
