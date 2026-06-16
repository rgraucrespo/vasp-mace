"""Tests for phonon force-constant tensor conventions."""

from __future__ import annotations

import unittest

import numpy as np

from vasp_mace.phonons import _force_constants_from_phonopy


class PhononForceConstantLayoutTests(unittest.TestCase):
    def test_phonopy_force_constants_convert_to_saved_layout(self) -> None:
        phonopy_fc = np.arange(2 * 2 * 3 * 3).reshape(2, 2, 3, 3)

        converted = _force_constants_from_phonopy(phonopy_fc)

        self.assertEqual(converted.shape, (2, 3, 2, 3))
        np.testing.assert_array_equal(converted, phonopy_fc.transpose(0, 2, 1, 3))
        self.assertEqual(converted[1, 2, 0, 1], phonopy_fc[1, 0, 2, 1])

    def test_phonopy_force_constants_reject_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            _force_constants_from_phonopy(np.zeros((2, 3, 3, 3)))


if __name__ == "__main__":
    unittest.main()
