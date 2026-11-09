"""Materials, layers, contacts and the 1D mesh.

Input units follow the conventions of the device-simulation literature
(cm^-3, cm^2/Vs, nm, eV) and are converted to SI internally.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .constants import CM3, CM2VS


@dataclass
class Material:
    name: str
    Eg: float              # band gap [eV]
    chi: float             # electron affinity [eV]
    eps_r: float           # relative permittivity
    Nc: float              # effective DOS, conduction band [cm^-3]
    Nv: float              # effective DOS, valence band [cm^-3]
    mu_n: float            # electron mobility [cm^2/Vs]
    mu_p: float            # hole mobility [cm^2/Vs]
    ND: float = 0.0        # donor density [cm^-3]
    NA: float = 0.0        # acceptor density [cm^-3]
    tau_n: float = 1e-6    # SRH electron lifetime [s]
    tau_p: float = 1e-6    # SRH hole lifetime [s]
    B_rad: float = 0.0     # radiative coefficient [cm^3/s]
    # absorption coefficient alpha(lambda_nm) -> [cm^-1]; None = transparent
    alpha: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass
class Layer:
    material: Material
    thickness_nm: float
    n_points: int = 120


@dataclass
class Contact:
    """Metal/TCO contact.

    W : work function [eV]; None = flat-band (ohmic) contact.
    Sn, Sp : surface recombination velocities [cm/s].
    """
    W: Optional[float] = None
    Sn: float = 1e7
    Sp: float = 1e7


@dataclass
class Interface:
    """Recombination at the interface between layer k and k+1 (midgap traps).

    Sn, Sp : interface recombination velocities [cm/s]; 0 disables it.
    """
    k: int
    Sn: float = 0.0
    Sp: float = 0.0


@dataclass
class Device:
    layers: List[Layer]
    left: Contact = field(default_factory=Contact)    # electron contact, light enters here
    right: Contact = field(default_factory=Contact)   # hole contact
    interfaces: List[Interface] = field(default_factory=list)
    delta_nm: float = 0.1      # gap between the two nodes of an interface
    cluster: float = 2.5       # mesh clustering towards layer boundaries

    def build_mesh(self):
        """Nodes, per-node layer index and the node pairs (a, b) at each interface."""
        xs, layer_id = [], []
        x0 = 0.0
        nl = len(self.layers)
        d = self.delta_nm * 1e-9
        for k, L in enumerate(self.layers):
            t = L.thickness_nm * 1e-9
            a = x0 + (d / 2 if k > 0 else 0.0)
            b = x0 + t - (d / 2 if k < nl - 1 else 0.0)
            s = np.linspace(-1.0, 1.0, L.n_points)
            frac = (np.tanh(self.cluster * s) / np.tanh(self.cluster) + 1.0) / 2.0
            xs.append(a + (b - a) * frac)
            layer_id.append(np.full(L.n_points, k))
            x0 += t
        x = np.concatenate(xs)
        lid = np.concatenate(layer_id)
        pairs = []
        start = 0
        for k, L in enumerate(self.layers[:-1]):
            start += L.n_points
            pairs.append((start - 1, start))   # last node of k, first node of k+1
        return x, lid, pairs

    def node_arrays(self, lid):
        """Material parameters per node, in SI units."""
        mats = [L.material for L in self.layers]

        def arr(attr, scale=1.0):
            return np.array([getattr(mats[k], attr) for k in lid], float) * scale

        return dict(
            Eg=arr("Eg"), chi=arr("chi"), eps_r=arr("eps_r"),
            Nc=arr("Nc", CM3), Nv=arr("Nv", CM3),
            mu_n=arr("mu_n", CM2VS), mu_p=arr("mu_p", CM2VS),
            ND=arr("ND", CM3), NA=arr("NA", CM3),
            tau_n=arr("tau_n"), tau_p=arr("tau_p"),
            B_rad=arr("B_rad", 1e-6),
        )
