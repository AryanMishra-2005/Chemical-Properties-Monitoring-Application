"""Headless validation of the chatbot intent router."""
import re
from mcp_server import get_chemical_phase, get_vapor_pressure

_NUM = r"[-+]?\d+(?:\.\d+)?"

_STRIP_PREFIXES = re.compile(
    r"^\s*(?:what(?:'s)? phase (?:is|are)|what is the|what(?:'s| is| are)?|"
    r"tell me|give me|calculate|compute|find|show|"
    r"is|are|phase of|vapor pressure of|"
    r"vapour pressure of|boiling point of|saturation pressure of|"
    r"pressure of|state of)\s+",
    re.IGNORECASE,
)
_TRAILING_NOISE = re.compile(
    r"\s*(?:a\s+)?(?:gas|liquid|solid|vapor|vapour|phase|state|at|"
    r"substance|compound|chemical)\s*$",
    re.IGNORECASE,
)
_KNOWN = [
    "carbon dioxide", "co2", "methane", "ethane", "propane", "butane",
    "pentane", "hexane", "ethanol", "methanol", "water", "benzene",
    "toluene", "nitrogen", "oxygen", "hydrogen", "ammonia", "acetone",
]

def _parse_and_dispatch(text):
    t = text.lower()
    nums = [float(m) for m in re.findall(_NUM, text)]
    cleaned = _STRIP_PREFIXES.sub("", text).strip()
    name_match = re.match(
        r"([a-zA-Z][a-zA-Z0-9 \-]*?)(?:\s+at\b|\s*@|\s*[-+]?\d|$)",
        cleaned, re.IGNORECASE)
    chemical = name_match.group(1).strip() if name_match else None
    if chemical:
        chemical = _TRAILING_NOISE.sub("", chemical).strip()
    if not chemical or len(chemical.split()) > 3:
        chemical = None
        for known in _KNOWN:
            if known in t:
                chemical = known
                break
    if not chemical:
        return ("I couldn't identify a chemical name.", True)
    if not nums:
        return (f"Found '{chemical}' but no temperature.", True)
    temp_c = nums[0]
    wants_vapor = any(kw in t for kw in ["vapor","vapour","saturation","boiling","psat"])
    if wants_vapor:
        return get_vapor_pressure(chemical, temp_c), False
    press_bar = nums[1] if len(nums) >= 2 else 1.01325
    return get_chemical_phase(chemical, temp_c, press_bar), False

QUESTIONS = [
    ("OK", "What phase is methane at -50 and 1 bar?"),
    ("OK", "Vapor pressure of ethanol at 78?"),
    ("OK", "Is water a gas at 200 and 1 bar?"),
    ("OK", "Phase of propane at 20, 8.5 bar?"),
    ("OK", "Vapor pressure of water at 100"),
    ("OK", "What is the boiling point of benzene at 25?"),
    ("OK", "phase of nitrogen at -150 and 10 bar"),
    ("OK",  "what phase is notachemical at 25"),   # tool returns graceful error string
    ("ERR", "tell me something"),
]

all_pass = True
for expected_tag, q in QUESTIONS:
    reply, err = _parse_and_dispatch(q)
    actual_tag = "ERR" if err else "OK "
    first = reply.splitlines()[0][:80]
    match = (expected_tag.strip() == actual_tag.strip())
    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
    print(f"[{status}][{actual_tag}] {q!r}")
    print(f"         {first}")
    print()

print("ALL PASS" if all_pass else "SOME TESTS FAILED")
