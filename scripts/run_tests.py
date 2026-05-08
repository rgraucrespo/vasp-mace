#!/usr/bin/env python3
"""Run the vasp-mace test suite.

Examples:
  python scripts/run_tests.py
  python scripts/run_tests.py --conda-env mace_env
  python scripts/run_tests.py --conda-env mace_env --with-examples
  python scripts/run_tests.py --conda-env mace_env --with-examples --example-set all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda-env",
        help=(
            "Run the suite through `conda run -n ENV python`. This is useful "
            "when `conda activate` is not available in the current shell."
        ),
    )
    parser.add_argument(
        "--with-examples",
        action="store_true",
        help="Run MACE-backed example smoke tests in addition to lightweight tests.",
    )
    parser.add_argument(
        "--example-set",
        choices=("quick", "all"),
        default="quick",
        help="Example smoke set to run when --with-examples is enabled.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MACE_MODEL_PATH"),
        help="MACE model checkpoint path for --with-examples.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("VASP_MACE_TEST_DEVICE", "auto"),
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device passed to vasp-mace during example smoke tests.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("VASP_MACE_TEST_DTYPE", "auto"),
        choices=("auto", "float32", "float64"),
        help="Dtype passed to vasp-mace during example smoke tests.",
    )
    parser.add_argument(
        "--timeout",
        default=os.environ.get("VASP_MACE_TEST_TIMEOUT", "600"),
        help="Per-example timeout in seconds.",
    )
    parser.add_argument(
        "--pattern",
        default="test*.py",
        help="unittest discovery pattern.",
    )
    return parser.parse_args()


def _relay_through_conda(args: argparse.Namespace) -> Optional[int]:
    if not args.conda_env:
        return None
    if os.environ.get("VASP_MACE_TEST_CONDA_RELAYED") == "1":
        return None

    forwarded = [str(Path(__file__).resolve())]
    if args.with_examples:
        forwarded.append("--with-examples")
    forwarded.extend(["--example-set", args.example_set])
    if args.model:
        forwarded.extend(["--model", args.model])
    forwarded.extend(["--device", args.device])
    forwarded.extend(["--dtype", args.dtype])
    forwarded.extend(["--timeout", str(args.timeout)])
    forwarded.extend(["--pattern", args.pattern])

    env = os.environ.copy()
    env["VASP_MACE_TEST_CONDA_RELAYED"] = "1"
    cmd = ["conda", "run", "-n", args.conda_env, "python", *forwarded]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    args = _parse_args()

    relayed = _relay_through_conda(args)
    if relayed is not None:
        return relayed

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    if args.with_examples:
        env["RUN_VASP_MACE_EXAMPLES"] = "1"
        env["VASP_MACE_EXAMPLE_SET"] = args.example_set
        env["VASP_MACE_TEST_DEVICE"] = args.device
        env["VASP_MACE_TEST_DTYPE"] = args.dtype
        env["VASP_MACE_TEST_TIMEOUT"] = str(args.timeout)
        if args.model:
            env["MACE_MODEL_PATH"] = args.model
        elif not env.get("MACE_MODEL_PATH"):
            print(
                "error: --with-examples requires --model or MACE_MODEL_PATH",
                file=sys.stderr,
            )
            return 2

    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(REPO_ROOT / "tests"),
        "-p",
        args.pattern,
        "-v",
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
