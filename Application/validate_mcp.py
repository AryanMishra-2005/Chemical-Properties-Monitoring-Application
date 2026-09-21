"""
validate_mcp.py
---------------
Exercises both tool functions from mcp_server.py without starting the MCP
transport layer.  Run with:  python validate_mcp.py
"""

# Import the bare functions directly (avoids spinning up the stdio server)
from mcp_server import get_chemical_phase, get_vapor_pressure

CASES = [
    # (tool, args, description)
    (get_chemical_phase, ("methane",  -50.0,   1.0),    "Methane gas at -50 °C, 1 bar"),
    (get_chemical_phase, ("water",     25.0,   1.01325),"Water liquid at 25 °C, 1 atm"),
    (get_chemical_phase, ("propane",   20.0,   8.5),    "Propane near two-phase at 20 °C, 8.5 bar"),
    (get_chemical_phase, ("water",    400.0, 230.0),    "Water supercritical at 400 °C, 230 bar"),
    (get_vapor_pressure, ("water",     100.0),          "Water Psat at 100 C (expect ~101325 Pa)"),
    (get_vapor_pressure, ("ethanol",    20.0),          "Ethanol Psat at 20 °C"),
    (get_vapor_pressure, ("7732-18-5",  25.0),          "Water by CAS at 25 °C"),
    (get_chemical_phase, ("notachemical", 25.0, 1.0),   "Invalid name (error path)"),
]

for fn, args, desc in CASES:
    print(f"\n{'='*60}")
    print(f"TEST : {desc}")
    print(f"CALL : {fn.__name__}{args}")
    print("-"*60)
    result = fn(*args)
    print(result)

print("\n\nAll tests completed.")
