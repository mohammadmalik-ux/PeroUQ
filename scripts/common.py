"""Shared helpers for the study scripts."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perouq import optics   # noqa: E402


def add_spectrum_args(ap):
    ap.add_argument("--spectrum", help="ASTM G173-03 file (csv/txt); the global-tilt column is used")
    ap.add_argument("--test-spectrum", action="store_true",
                    help="use the 5778 K black-body test spectrum (plumbing checks only, NOT results)")


def load_spectrum(args):
    """Returns (lam_nm, E, tag).  tag is '' for AM1.5G and a warning for the test spectrum."""
    if args.spectrum:
        lam, E = optics.load_astm_g173(args.spectrum, "global")
        P = np.trapezoid(E, lam)
        if not 990 < P < 1010:
            raise SystemExit(f"spectrum integrates to {P:.1f} W/m2; expected ~1000 for AM1.5G global")
        return lam, E, ""
    if args.test_spectrum:
        lam, E = optics.blackbody_test_spectrum()
        return lam, E, "TEST SPECTRUM - NOT RESULTS"
    raise SystemExit("give --spectrum ASTMG173.csv (or --test-spectrum for a plumbing check)")
