"""Regression tests for the IVDW (DFT-D3 dispersion) plumbing.

These guard against the prior bug where ``MACECalculator(dispersion=True)`` was
silently dropped — the kwarg was swallowed by ``Calculator.__init__`` and D3
never ran, so ``IVDW=12`` produced bit-identical results to ``IVDW=0``.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from vasp_mace.incar import parse_incar


def _torch_dftd_available() -> bool:
    try:
        import torch_dftd  # noqa: F401
    except ImportError:
        return False
    return True


def _write_incar(tmp: Path, body: str) -> str:
    path = tmp / "INCAR"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return str(path)


class IVDWIncarParsingTests(unittest.TestCase):
    def test_accepts_d3_zero_and_bj(self) -> None:
        for ivdw in (11, 12):
            with self.subTest(ivdw=ivdw), tempfile.TemporaryDirectory() as td:
                cfg = parse_incar(_write_incar(Path(td), f"IVDW = {ivdw}\n"))
                self.assertEqual(cfg.IVDW, ivdw)

    def test_rejects_atm_three_body(self) -> None:
        for ivdw in (13, 14):
            with self.subTest(ivdw=ivdw), tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(ValueError, "ATM"):
                    parse_incar(_write_incar(Path(td), f"IVDW = {ivdw}\n"))

    def test_rejects_unknown_ivdw(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "IVDW=9 is not supported"):
                parse_incar(_write_incar(Path(td), "IVDW = 9\n"))


@unittest.skipUnless(_torch_dftd_available(), "torch_dftd not installed")
class IVDWDispersionPlumbingTests(unittest.TestCase):
    """Confirm the wrapped calculator actually adds D3 to a stub calculator.

    We bypass the real MACE checkpoint (heavy, requires GPU on most setups) by
    feeding ``_wrap_with_d3`` an EMT stub. The test only cares that the wrapped
    calculator returns energies and forces that differ from the bare stub —
    proof that D3 is being summed in.
    """

    def _build_pair(self):
        from ase import Atoms
        from ase.calculators.emt import EMT
        from vasp_mace.mace_loader import _wrap_with_d3

        # Two Cu atoms 3.5 Å apart — EMT supports Cu, and D3 still adds a
        # non-zero correction whatever the underlying potential is.
        atoms_bare = Atoms("Cu2", positions=[(0, 0, 0), (3.5, 0, 0)], pbc=False)
        atoms_d3 = atoms_bare.copy()
        atoms_bare.calc = EMT()
        atoms_d3.calc = _wrap_with_d3(EMT(), "cpu", "float64", "bj")
        return atoms_bare, atoms_d3

    def test_wrapped_calc_is_sum_calculator(self) -> None:
        from ase.calculators.emt import EMT
        from ase.calculators.mixing import SumCalculator
        from vasp_mace.mace_loader import _wrap_with_d3

        wrapped = _wrap_with_d3(EMT(), "cpu", "float64", "bj")
        self.assertIsInstance(wrapped, SumCalculator)

    def test_d3_changes_energy_and_forces(self) -> None:
        atoms_bare, atoms_d3 = self._build_pair()
        e_bare = atoms_bare.get_potential_energy()
        e_d3 = atoms_d3.get_potential_energy()
        # D3 attractive correction at 3.5 Å between Ar atoms is non-trivial
        # (~meV scale). Anything above 1e-6 eV proves the wrapping is live.
        self.assertGreater(abs(e_d3 - e_bare), 1.0e-6)

        f_bare = atoms_bare.get_forces()
        f_d3 = atoms_d3.get_forces()
        max_df = float(((f_d3 - f_bare) ** 2).sum(axis=1).max() ** 0.5)
        self.assertGreater(max_df, 1.0e-6)


class IVDWLoadCalcSignatureTests(unittest.TestCase):
    def test_load_calc_rejects_unsupported_ivdw(self) -> None:
        from vasp_mace.mace_loader import load_calc

        # Use a path that won't exist; we want the IVDW guard to fire before
        # the model file is inspected.
        with self.assertRaisesRegex(ValueError, "IVDW=13 is not supported"):
            load_calc("/nonexistent/model.model", ivdw=13)


if __name__ == "__main__":
    unittest.main()
