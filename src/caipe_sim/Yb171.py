r"""171Yb+ metadata; published inputs are catalogued in docs/physics-references.md.

Wavelength context: PRA 83, 013406 (2011).
P lifetime: PRA 80, 022502 (2009); D lifetime: PRA 61, 022507 (2000).
Auxiliary lifetime: J. Phys. B 26, L701 (1993).
Branching and auxiliary HFS: PRA 76, 052314 (2007).
S HFS: IEEE Trans. UFFC 44, 344 (1997); P HFS: PRA 49, 3351 (1994).
Magnetic-moment convention: arXiv:2509.04416 (neutral Yb; not ion level data).
D splitting is 0.86375 GHz; metadata stores A/h=0.431875 GHz (splitting/2).
"""

import numpy as np
import scipy.constants as consts

# Minimal local stand-ins so we do not depend on atomic_physics externally.
class Level:
    def __init__(self, n: int, S: float, L: float, J: float):
        self.n = n
        self.S = S
        self.L = L
        self.J = J


class LevelData:
    def __init__(self, level: Level, Ahfs: float, Bhfs: float, g_J: float, g_I: float):
        self.level = level
        self.Ahfs = Ahfs
        self.Bhfs = Bhfs
        self.g_J = g_J
        self.g_I = g_I


class Transition:
    def __init__(self, lower: Level, upper: Level, einstein_A: float, frequency: float):
        self.lower = lower
        self.upper = upper
        self.einstein_A = einstein_A
        self.frequency = frequency

# Transition data 

# Lifetimes: published sources listed in the module docstring.
tau_P12   = 8.12e-9       # 6p 2P1/2
tau_D32   = 52.7e-3       # 5d 2D3/2
tau_D3212  = 37.7e-9      # 3D[3/2]1/2  

# Branching: PRA 76, 052314 (2007); retain the two-branch complement.
BR_P12_to_S12 = 1.0 - 0.00501
BR_P12_to_D32 = 0.00501

# Einstein A coefficients for each transition (Using formula: A = 1/tau * (branching ratio)  )
A_P12_total   = 1.0 / tau_P12                 
A_369_to_S12  = BR_P12_to_S12 * A_P12_total   # P1/2 -> S1/2
A_2438_to_D32 = BR_P12_to_D32 * A_P12_total   # P1/2 -> D3/2
A_D32_to_S12  = 1.0 / tau_D32                 # D3/2 -> S1/2 

# Angular frequencies (Using formula: ω = 2πc/λ)
omega_369  = 2 * np.pi * consts.c / (369.5262e-9)   # S1/2 <-> P1/2
omega_2438 = 2 * np.pi * consts.c / (2.438e-6)      # D3/2 <-> P1/2
omega_435  = 2 * np.pi * consts.c / (435.5e-9)      # S1/2 <-> D3/2 (E2; unused transition metadata)

# I for Yb-171 is always 1/2 
I = 0.5

# Hyperfine metadata: source and A-versus-splitting caveat in physics-references.md.
nu_S12 =   12.642812118466e9   # F=1 <-> F=0  (6s 2S1/2)
nu_P12 =   2.105e9    # F=1 <-> F=0  (6p 2P1/2)
nu_D32 =   0.86375e9    # F=2 <-> F=1  (5d 2D3/2)
nu_D3212 = 2.2095e9    # F=0 <-> F=1  (3D[3/2]1/2)


def lande_g_factor(level: Level) -> float:
    """Compute Landé g-factor for a level."""
    J = level.J
    if J == 0:
        raise ValueError("Landé g-factor undefined for J = 0.")
    S = level.S
    L = level.L
    return 1 + ((J * (J + 1) - L * (L + 1) + S * (S + 1)) / (2 * J * (J + 1)))

class Yb171Factory:
    r""":class:`~atomic_physics.core.AtomFactory` for :math:`^{171}\mathrm{Yb}^+`.

    Attributes:
        S12, P12, D32: the retained S1/2, P1/2 and D3/2 ionic manifolds.
        D3212: auxiliary 3D[3/2]1/2 metadata with J=1/2, absent from the
            evolved basis. Its legacy S,L fields are not an LS configuration
            for calculating couplings; see docs/physics-references.md.
        ground_level: alias for S12.
        shelf: legacy compatibility alias for D3212.
    """

    S12: Level = Level(n=6, S=1 / 2, L=0, J=1 / 2)
    P12: Level = Level(n=6, S=1 / 2, L=1, J=1 / 2)
    D32: Level = Level(n=5, S=1 / 2, L=2, J=3 / 2)
    D3212: Level = Level(n=5, S=1 / 2, L=2, J=1 / 2)

    ground_level: Level = S12
    shelf: Level = D3212

    def __init__(self):
        level_data = (
            LevelData(
                level=self.S12,
                Ahfs=nu_S12 * consts.h,  
                Bhfs=0,
                g_J=lande_g_factor(self.S12),  
                g_I=-0.0005362,
            ),
            LevelData(
                level=self.P12,
                Ahfs=nu_P12 * consts.h,  
                Bhfs=0,
                g_J=lande_g_factor(self.P12),
                g_I=-0.0005362,
            ),
            LevelData(
                level=self.D32,
                Ahfs=(nu_D32 / 2) * consts.h,  # A = splitting/2 for I=1/2,J=3/2.
                Bhfs=0,  
                g_J=lande_g_factor(self.D32),
                g_I=-0.0005362,
            ),
        )
        

        
        transitions = {
            "369": Transition(
                lower=self.S12,
                upper=self.P12,
                einstein_A=A_369_to_S12,   
                frequency=omega_369,       
            ),

            "2438": Transition(
                lower=self.D32,
                upper=self.P12,
                einstein_A=A_2438_to_D32,
                frequency=omega_2438,
            ),

            "435": Transition(
                lower=self.S12,
                upper=self.D32,
                einstein_A=A_D32_to_S12,   
                frequency=omega_435,       
            ),
            
        }

        self.nuclear_spin = 1 / 2
        self.level_data = level_data
        self.transitions = transitions


Yb171 = Yb171Factory()
r""" :math:`^{171}\mathrm{Yb}^+` atomic structure. """
