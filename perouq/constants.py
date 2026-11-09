"""Physical constants (SI)."""

q = 1.602176634e-19       # elementary charge [C]
kB = 1.380649e-23         # Boltzmann constant [J/K]
eps0 = 8.8541878128e-12   # vacuum permittivity [F/m]
h_planck = 6.62607015e-34 # Planck constant [J s]
c_light = 2.99792458e8    # speed of light [m/s]

CM3 = 1e6    # 1 cm^-3  -> m^-3
CM2VS = 1e-4 # 1 cm^2/Vs -> m^2/Vs


def thermal_voltage(T=300.0):
    """kT/q in volts (numerically equal to kT in eV)."""
    return kB * T / q
