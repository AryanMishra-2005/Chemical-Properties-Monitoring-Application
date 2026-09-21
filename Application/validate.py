"""Headless validation: runs the full compute pipeline for all 4 chemicals."""
import sqlite3
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, FlashPureVLS
from thermo.eos_mix import PRMIX

con = sqlite3.connect("chemicals.db")
con.row_factory = sqlite3.Row
rows = con.execute("SELECT * FROM chemicals").fetchall()
con.close()

for r in rows:
    chem = dict(r)
    name = chem["name"]
    print(f"--- {name} ---")

    consts, props = ChemicalConstantsPackage.from_IDs([chem["cas"]])
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
    _, props = ChemicalConstantsPackage.from_IDs([chem["cas"]])

    eos_kw = dict(Tcs=[chem["Tc"]], Pcs=[chem["Pc"]], omegas=[chem["omega"]])
    gas   = CEOSGas(PRMIX, eos_kwargs=eos_kw, HeatCapacityGases=props.HeatCapacityGases)
    liq   = CEOSLiquid(PRMIX, eos_kwargs=eos_kw, HeatCapacityGases=props.HeatCapacityGases)
    flash = FlashPureVLS(consts, props, gas=gas, liquids=[liq], solids=[])

    res  = flash.flash(T=298.15, P=101325.0)
    bulk = res.bulk if hasattr(res, "bulk") else res

    print(f"  Phase : {res.phase}")
    print(f"  Z     : {bulk.Z():.5f}")
    print(f"  rho   : {bulk.rho_mass():.4f} kg/m3")
    print(f"  H     : {bulk.H()/1000:.4f} kJ/mol")
    print(f"  Cp    : {bulk.Cp()/1000:.5f} kJ/mol.K")
    print()

print("ALL OK")
