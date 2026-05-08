"""Backward-compatible re-export shim.

All public symbols are now defined in the focused sub-modules:
  io_poscar  — POSCAR / CONTCAR / XDATCAR
  io_outcar  — OSZICAR / OUTCAR (relaxation, single-point, MD)
  io_xml     — vasprun.xml

Import from those modules directly in new code.
"""

from .io_poscar import (
    read_poscar,
    write_contcar,
    write_xdatcar_header,
    append_xdatcar_frame,
)
from .io_outcar import (
    write_oszicar,
    write_outcar,
    write_outcar_tail,
    write_outcar_like,
    write_md_outcar_header,
    append_md_outcar_step,
)
from .io_xml import (
    write_relax_vasprun_xml,
    write_single_vasprun_xml,
)

__all__ = [
    "read_poscar",
    "write_contcar",
    "write_xdatcar_header",
    "append_xdatcar_frame",
    "write_oszicar",
    "write_outcar",
    "write_outcar_tail",
    "write_outcar_like",
    "write_md_outcar_header",
    "append_md_outcar_step",
    "write_relax_vasprun_xml",
    "write_single_vasprun_xml",
]
