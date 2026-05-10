"""Reader and writer for the VASP-compatible ``ML_HEAT`` file.

The ``ML_HEAT`` file is the per-step heat-flux output produced by VASP's
``ML_LHEAT`` machine-learning workflow. Each MD step contributes one line:

    NSTEP=         1 QXYZ=  0.36329995E-03 -0.18158424E-03 -0.89885493E-03

Heat-flux values are stored in ``eV·Å·fs⁻¹``. The writer in this module emits a
fixed-width, scientific-notation form that matches the VASP layout closely; the
reader is permissive about whitespace and accepts both Python (``E``) and
Fortran (``D``) exponent markers so it can ingest files produced by either
toolchain.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional, Tuple, Union

import numpy as np


_LINE_RE = re.compile(
    r"^\s*NSTEP\s*=\s*(?P<step>[-+]?\d+)\s+"
    r"QXYZ\s*=\s*"
    r"(?P<qx>\S+)\s+(?P<qy>\S+)\s+(?P<qz>\S+)\s*$"
)


def _format_line(step: int, qxyz: np.ndarray) -> str:
    qx, qy, qz = float(qxyz[0]), float(qxyz[1]), float(qxyz[2])
    return f"NSTEP={int(step):10d} QXYZ= {qx:16.8E} {qy:16.8E} {qz:16.8E}\n"


def _parse_fortran_float(token: str) -> float:
    """Parse a numeric token, tolerating Fortran-style ``D`` exponents."""
    cleaned = token.replace("D", "E").replace("d", "e")
    return float(cleaned)


def write_ml_heat(
    path: str,
    steps: Iterable[int],
    qxyz: np.ndarray,
) -> None:
    """Write a complete ``ML_HEAT`` file in one call.

    Parameters
    ----------
    path
        Destination file path. Existing files are overwritten.
    steps
        Iterable of integer MD step indices.
    qxyz
        Array of heat-flux vectors in ``eV·Å·fs⁻¹``. Must have shape ``(n, 3)``
        where ``n`` is the number of step entries supplied.

    Raises
    ------
    ValueError
        If ``qxyz`` is not 2-D with three columns or its row count does not
        match ``steps``.
    """
    q = np.asarray(qxyz, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"qxyz must have shape (n, 3); got {q.shape}")
    step_list = [int(s) for s in steps]
    if len(step_list) != q.shape[0]:
        raise ValueError(
            f"steps has {len(step_list)} entries but qxyz has {q.shape[0]} rows"
        )
    with open(path, "w") as fh:
        for s, row in zip(step_list, q):
            fh.write(_format_line(s, row))


def read_ml_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an ``ML_HEAT`` file produced by vasp-mace or VASP.

    Parameters
    ----------
    path
        Path to the ``ML_HEAT`` file.

    Returns
    -------
    tuple of numpy.ndarray
        ``(steps, qxyz)`` where ``steps`` has shape ``(n,)`` and dtype
        ``int64``, and ``qxyz`` has shape ``(n, 3)`` and dtype ``float64``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a non-blank line cannot be parsed as ``NSTEP=… QXYZ=…``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ML_HEAT not found: {os.path.abspath(path)}")

    steps: list[int] = []
    qxyz: list[tuple[float, float, float]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if m is None:
                raise ValueError(
                    f"{path}:{lineno}: unrecognised ML_HEAT line: {line.rstrip()!r}"
                )
            steps.append(int(m.group("step")))
            qxyz.append(
                (
                    _parse_fortran_float(m.group("qx")),
                    _parse_fortran_float(m.group("qy")),
                    _parse_fortran_float(m.group("qz")),
                )
            )

    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(qxyz, dtype=np.float64).reshape(-1, 3),
    )


class MLHeatWriter:
    """Streaming writer for the ``ML_HEAT`` file.

    Designed for use during MD: the writer keeps a file handle open between
    steps and flushes periodically so a long run leaves a usable file even if
    interrupted. Use as a context manager or call :meth:`close` explicitly.

    Parameters
    ----------
    path
        Destination file path. Defaults to ``"ML_HEAT"`` in the current
        directory.
    mode
        File-open mode. ``"w"`` (default) truncates an existing file; ``"a"``
        appends.
    flush_every
        Flush the underlying file handle every ``flush_every`` writes. The
        default of ``100`` is small enough that interrupted runs lose at most
        ``flush_every`` recent steps without paying a per-step ``fsync``.
    """

    def __init__(
        self,
        path: str = "ML_HEAT",
        mode: str = "w",
        flush_every: int = 100,
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a'; got {mode!r}")
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1; got {flush_every}")
        self._path = path
        self._mode = mode
        self._flush_every = flush_every
        self._fh: Optional[object] = open(path, mode)
        self._n_written = 0

    @property
    def path(self) -> str:
        return self._path

    def write(self, step: int, qxyz: Union[np.ndarray, Iterable[float]]) -> None:
        """Append one ``NSTEP=… QXYZ=…`` line to the file.

        Parameters
        ----------
        step
            MD step index.
        qxyz
            Heat-flux vector ``[qx, qy, qz]`` in ``eV·Å·fs⁻¹``. Must have
            three components.

        Raises
        ------
        RuntimeError
            If the writer has already been closed.
        ValueError
            If ``qxyz`` does not have exactly three components.
        """
        if self._fh is None:
            raise RuntimeError("MLHeatWriter is closed")
        q = np.asarray(qxyz, dtype=float).reshape(-1)
        if q.size != 3:
            raise ValueError(f"qxyz must have 3 components; got shape {q.shape}")
        self._fh.write(_format_line(int(step), q))
        self._n_written += 1
        if self._n_written % self._flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file. Safe to call repeatedly."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "MLHeatWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
