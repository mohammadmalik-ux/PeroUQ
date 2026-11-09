"""Integrity checks for the literature parameter database."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perouq import params   # noqa: E402

UNITS = {"Eg": "eV", "chi": "eV", "eps_r": "-", "Nc": "cm^-3", "Nv": "cm^-3",
         "mu_n": "cm^2/Vs", "mu_p": "cm^2/Vs", "ND": "cm^-3", "NA": "cm^-3", "Nt": "cm^-3",
         "sigma": "cm^2", "vth": "cm/s", "W": "eV", "S": "cm/s"}
BOUNDS = {"Eg": (0.5, 5), "chi": (1, 6), "eps_r": (1, 100), "W": (3, 6)}


def test_decisions_are_applied():
    t = params.table("MAPbI3")
    assert t["eps_r"]["max"] <= 10.0
    assert set(t["mu_n"]["sources"]) == {"Herz2017"}
    assert set(t["mu_p"]["sources"]) == {"Herz2017"}
    raw = params.table("MAPbI3", params.load_rows(decisions=False))
    assert raw["eps_r"]["max"] == 30.0          # raw literature data are kept intact
    for d in params.load_decisions():
        assert d["reason"] and d["date"], d


def test_every_row_has_a_known_source_and_unit():
    rows, src = params.load_rows(decisions=False), params.load_sources()
    for r in rows:
        assert r["source"] in src, r
        assert UNITS[r["param"]] == r["unit"], r
        lo, hi = BOUNDS.get(r["param"], (0, np.inf))
        assert lo <= r["value"] <= hi, r


def test_sources_have_doi_or_url():
    for s in params.load_sources().values():
        assert s["doi"] or s["url"], s


def test_materials_build_and_sample_inside_ranges():
    rng = np.random.default_rng(0)
    for m in ("MAPbI3", "TiO2", "SnO2", "ZnO", "CdS"):
        params.material(m)
        rg = params.ranges(m)
        for _ in range(20):
            k = sum(1 for r in rg.values() if r[0] != r[1])
            vals, names = params.sample_values(m, rng.random(k))
            for p in names:
                lo, hi, _ = rg[p]
                assert lo * (1 - 1e-12) <= vals[p] <= hi * (1 + 1e-12), (m, p, vals[p])
            params.material(m, vals)


if __name__ == "__main__":
    test_decisions_are_applied()
    test_every_row_has_a_known_source_and_unit()
    test_sources_have_doi_or_url()
    test_materials_build_and_sample_inside_ranges()
    print("database: OK")
    print(params.summary_markdown())


def test_min_spread_keeps_wider_ranges():
    """The minimum-uncertainty rule widens only ranges narrower than the minimum."""
    from perouq import params
    from perouq.params import DEFAULT_MIN_SPREAD
    r = params.ranges("MAPbI3", min_spread=DEFAULT_MIN_SPREAD)
    assert r["eps_r"][:2] == (6.5, 10.0)                      # author's bounds kept
    assert abs(r["Eg"][0] - 1.45) < 1e-9 and abs(r["Eg"][1] - 1.65) < 1e-9   # 0.1 eV wide -> +/-0.1 eV
