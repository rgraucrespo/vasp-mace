"""Backend-independent heat-flux calculator interface.

Concrete heat-flux backends (currently only :mod:`vasp_mace.heat.mace_unfolded`)
are imported lazily through :func:`make_heat_flux_calculator` so the default
``vasp-mace`` install does not pull in heavy optional dependencies.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


class HeatFluxCalculator:
    """Abstract heat-flux calculator.

    Subclasses must implement :meth:`compute` and return a heat-flux vector
    ``[qx, qy, qz]`` in ``eV·Å·fs⁻¹`` for a given ASE ``Atoms`` object plus its
    velocity array.
    """

    def compute(self, atoms: Any, velocities: np.ndarray) -> np.ndarray:
        raise NotImplementedError


INSTALL_HINT = (
    "ML_LHEAT requires the optional heat-flux dependencies: comms and "
    "mace-unfolded. Install with: pip install "
    "git+https://github.com/sirmarcel/comms.git "
    "git+https://github.com/pulgon-project/mace-unfolded.git"
)


def make_heat_flux_calculator(
    model_path: str,
    settings: Optional[Mapping[str, Any]] = None,
) -> HeatFluxCalculator:
    """Build a heat-flux calculator for a given MACE model.

    Parameters
    ----------
    model_path
        Path to a MACE ``.model`` checkpoint.
    settings
        Optional mapping of backend settings. Recognised keys:

        - ``backend`` (default ``"mace_unfolded"``).
        - ``device``, ``dtype`` — see :func:`vasp_mace.mace_loader.load_calc`.
        - ``flux_type`` — currently only ``"potential"`` is supported.
        - ``forward`` — forward- vs reverse-mode autodiff in mace-unfolded.
        - ``torch_model`` — optional already-loaded raw MACE torch model,
          avoiding a second checkpoint load.

        Unknown keys are forwarded to the chosen backend, which raises
        ``TypeError`` on unrecognised arguments.

    Returns
    -------
    HeatFluxCalculator
        A backend-specific instance ready for use in MD.

    Raises
    ------
    ImportError
        If the requested backend's optional dependencies are not installed.
    ValueError
        If ``backend`` is not recognised.
    """
    settings = dict(settings or {})
    backend = settings.pop("backend", "mace_unfolded")
    if backend == "mace_unfolded":
        try:
            from .mace_unfolded import MACEUnfoldedHeatFluxCalculator
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc
        return MACEUnfoldedHeatFluxCalculator(model_path, **settings)
    raise ValueError(
        f"Unknown heat-flux backend {backend!r}; expected 'mace_unfolded'"
    )
