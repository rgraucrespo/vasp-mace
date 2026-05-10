"""Heat-flux file I/O and calculator interface for vasp-mace.

This subpackage handles the VASP-compatible ``ML_HEAT`` output file that vasp-mace
writes during MACE MD when ``ML_LHEAT = .TRUE.``, and the backend-independent
heat-flux calculator interface used to compute the per-step flux. The default
backend (:class:`vasp_mace.heat.mace_unfolded.MACEUnfoldedHeatFluxCalculator`)
wraps `mace-unfolded <https://github.com/pulgon-project/mace-unfolded>`_ and is
opt-in via ``pip install vasp-mace[heat]``.

Post-processing of ``ML_HEAT`` into thermal conductivity is intentionally out
of scope; pass the file to ``sportran``
(https://www.sciencedirect.com/science/article/abs/pii/S0010465522001898) for
Green-Kubo / cepstral analysis.
"""

from .heat_flux import HeatFluxCalculator, make_heat_flux_calculator
from .ml_heat import MLHeatWriter, read_ml_heat, write_ml_heat

__all__ = [
    "HeatFluxCalculator",
    "MLHeatWriter",
    "make_heat_flux_calculator",
    "read_ml_heat",
    "write_ml_heat",
]
