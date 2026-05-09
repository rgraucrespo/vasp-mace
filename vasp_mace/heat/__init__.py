"""Heat-flux file I/O for vasp-mace.

This subpackage handles the VASP-compatible ``ML_HEAT`` output file that vasp-mace
writes during MACE MD when ``ML_LHEAT = .TRUE.``. Post-processing of the heat flux
into thermal conductivity is intentionally out of scope; users should pass the
``ML_HEAT`` file to ``sportran``
(https://www.sciencedirect.com/science/article/abs/pii/S0010465522001898) for
Green-Kubo / cepstral analysis.
"""

from .ml_heat import MLHeatWriter, read_ml_heat, write_ml_heat

__all__ = ["MLHeatWriter", "read_ml_heat", "write_ml_heat"]
