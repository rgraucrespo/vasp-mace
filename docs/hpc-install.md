# Installing vasp-mace on a shared HPC cluster

The [README installation steps](../README.md#installation) assume a machine you
control — a laptop, or a workstation with a GPU. On a shared HPC cluster, three
things are different, and each one can turn `pip install vasp-mace` into a wall
of compiler errors:

- the module system exports compiler variables into your shell,
- the compute image may be old enough that recent wheels no longer match it,
- your home directory is quota-limited and the login node's `/tmp` is small.

This guide is written around **UCL's Young**, where every command below is
known to work. If you are on a different cluster, follow the quick start
anyway and see [Adapting to another cluster](#adapting-to-another-cluster) for
the parts you need to substitute.

---

## Quick start (Young)

Run this in a **fresh login shell**, top to bottom. It assumes you have
Miniconda or Miniforge installed in your home directory.

```bash
# 1. Clean the shell. Young's default modules export CC=icc / CXX=icpc,
#    which will be handed to any package that gets built from source.
module purge
unset CC CXX FC F77 F90 LDSHARED CFLAGS CXXFLAGS LDFLAGS
source $HOME/miniconda3/etc/profile.d/conda.sh

# Sanity check: a home-directory conda, and no compiler variables.
which conda && echo "CC=$CC CXX=$CXX"

# 2. Create the environment from conda-forge.
conda create -n vasp_mace_env -c conda-forge --override-channels python=3.11 -y
conda activate vasp_mace_env
python -m pip install -U pip

# 3. Keep build temp and pip cache off the small, shared login-node /tmp.
mkdir -p $HOME/Scratch/tmp $HOME/Scratch/.pip-cache
export TMPDIR=$HOME/Scratch/tmp
export PIP_CACHE_DIR=$HOME/Scratch/.pip-cache

# 4. Install the compiled dependencies with conda, not pip.
#    Young has no GPUs, so take the CPU torch build.
conda install -c conda-forge --override-channels numpy pytorch-cpu ase -y
python -c "import numpy, torch, ase; print(numpy.__version__, torch.__version__, ase.__version__)"

# 5. One mace-torch dependency ships no wheel at all. It is pure Python,
#    so it is safe to build; install it before the guarded step below.
pip install python-hostlist

# 6. Install vasp-mace, wheels only.
pip install --only-binary=:all: vasp-mace

# 7. Verify from a directory that is NOT a vasp-mace checkout.
cd ~
which vasp-mace
vasp-mace --help
```

Then download a model and set `MACE_MODEL_PATH` as described under
[Model checkpoint](../README.md#model-checkpoint) — put the `.model` file on
Scratch rather than in your quota-limited home:

```bash
cd $HOME/Scratch
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model
echo 'export MACE_MODEL_PATH=$HOME/Scratch/2024-01-07-mace-128-L2_epoch-199.model' >> ~/.bashrc
```

### Development install

To work on the code rather than just run it, clone the repository and replace
step 6 with an editable install. The dependencies are already in place from
steps 4–5, so skip the resolver entirely:

```bash
git clone https://github.com/rgraucrespo/vasp-mace.git $HOME/soft/vasp-mace
pip install -e $HOME/soft/vasp-mace --no-deps

cd ~ && which vasp-mace && vasp-mace --help
python -m pytest -q $HOME/soft/vasp-mace
```

Edits to the checkout then take effect immediately, with no reinstall.

---

## What each step is defending against

Four distinct failures, each with the error message it produces. Match the
symptom, apply the fix.

### 1. Compiler variables leaking in from the module system

**Symptom** — pip tries to build a package and the compiler errors are clearly
not from GCC:

```
../numpy/_core/src/common/npy_atomic.h(105): warning #2330: argument of type
"void *_Atomic *" is incompatible with parameter of type "volatile void *"
...
ninja: build stopped: subcommand failed.
```

Numbered `#2330`-style diagnostics come from Intel's `icc`/`icx`. numpy's Meson
build is not supported under the Intel compilers and will fail.

**Cause** — the cluster's default module set exports `CC`, `CXX` and friends.
pip inherits them for every source build. `conda activate` does not clear them,
because they live in the shell, not the environment.

**Fix** — `module purge` and `unset CC CXX FC F77 F90 LDSHARED CFLAGS CXXFLAGS
LDFLAGS`, once per shell, *before* installing. Confirm with `echo "CC=$CC"`.

Note this is only ever a *secondary* failure: it fires because something else
already forced a source build. Fixing it stops the crash; fixing failure 2
stops the build being attempted at all.

### 2. No matching wheel, so pip falls back to compiling

**Symptom** — pip downloads a `.tar.gz` instead of a `.whl` for a large
compiled package such as numpy, scipy or torch, and starts a build that takes
minutes and then fails.

**Cause** — no wheel on PyPI matched the interpreter *and* the platform. On a
current Python this is almost always the platform half: wheels are tagged
against a minimum glibc (`manylinux_2_17`, `manylinux_2_28`, …) and projects
periodically raise that floor. A cluster image older than the new floor stops
matching, silently, and pip's only remaining option is the source
distribution.

**Diagnose** — these three lines identify it:

```bash
ldd --version | head -1                      # the system glibc
pip debug --verbose | grep -c manylinux_2_28 # 0 means that tag is unusable here
pip install --only-binary=:all: --dry-run numpy
```

The third is the decisive one: with `--only-binary` pip refuses to compile and
instead states plainly that no distribution matched.

**Fix** — install the compiled packages from **conda-forge** instead of PyPI
(step 4). conda-forge builds against an old glibc baseline and does not depend
on PyPI's tagging decisions, so they install regardless. Pinning an older
version on PyPI (`pip install "numpy<2.3"`) also works, but it has to be
repeated for every affected package and redone at every upgrade.

### 3. A dependency that publishes no wheel at all

**Symptom** — `--only-binary=:all:` causes a resolution failure that names a
package, after pip has visibly backtracked through old vasp-mace releases:

```
The conflict is caused by:
    mace-torch 0.3.16 depends on python-hostlist
    mace-torch 0.3.15 depends on python-hostlist

Additionally, some packages in these conflicts have no matching distributions
available for your environment:
    python-hostlist
ERROR: ResolutionImpossible
```

**Cause** — `python-hostlist` (pulled in by `mace-torch`) is distributed as an
sdist only, for every platform. `--only-binary=:all:` forbids sdists, so pip
cannot use it and works backwards looking for a vasp-mace old enough not to
need it.

**Fix** — install it on its own first, then rerun the guarded command:

```bash
pip install python-hostlist
pip install --only-binary=:all: vasp-mace
```

This is safe because `python-hostlist` is pure Python: building it invokes no
compiler, so failure 1 cannot bite. Apply the same treatment to any other
pure-Python package named this way. Keep the guard on everything else — it is
your tripwire against pip quietly starting a source build.

### 4. A stale editable install shadowing the real one

**Symptom** — the install reports success, but:

```
$ vasp-mace --help
-bash: vasp-mace: command not found
```

while `pip show vasp-mace` reports the correct version, with a `Location:`
pointing at a source checkout rather than `site-packages`.

**Cause** — an earlier development install (typically legacy `setup.py
develop`) is still registered. pip sees the requirement as already satisfied,
installs only the missing dependencies, and never unpacks the wheel that would
have provided the `bin/vasp-mace` launcher. It survives environment rebuilds
because part of it lives in the checkout (`vasp_mace.egg-info/`), not in the
environment.

**Fix** — clear it out, then reinstall:

```bash
cd ~
pip uninstall -y vasp-mace
pip show vasp-mace                 # must report "not found"
rm -rf <checkout>/vasp_mace.egg-info <checkout>/build
grep -rl "<checkout>" $CONDA_PREFIX/lib/python3.11/site-packages/*.pth 2>/dev/null
```

If that `grep` names a `.pth` file (often `easy-install.pth`), delete the
offending line from it. Then install normally, or use `pip install -e` for a
modern editable install, which does create the launcher correctly.

Always verify from **outside** the checkout. With your shell sitting in a
vasp-mace source directory, `import vasp_mace` resolves to the current
directory and will succeed no matter what is really installed.

---

## Adapting to another cluster

The structure of the quick start holds anywhere; four things are site-specific.

| Young | What to substitute |
| --- | --- |
| `module purge` | Same on Lmod and environment-modules. If `conda` disappears afterwards, re-source the conda hook — or your site ships conda *as* a module, in which case reload it. |
| `CC=icc`, `CXX=icpc` | Check with `env \| grep -E '^(CC\|CXX\|F[0-9C])='`. Unset whatever is set; the names above cover the usual ones. |
| `$HOME/Scratch` | Your site's scratch path (`$SCRATCH`, `/work/$USER`, …). Used for `TMPDIR`, the pip cache, and model files. |
| SGE job script | Your scheduler's preamble and its core-count variable (see below). |

Before installing anywhere new, run the diagnostics from failure 2 — glibc,
usable wheel tags, and a `--dry-run` — and `echo "CC=$CC CXX=$CXX"` from
failure 1. Between them they predict which of the four failures you are about
to hit.

One decision does not carry over: **`pytorch-cpu`**. Young has no GPUs, so the
CPU build is correct there and avoids pulling roughly 2.5 GB of unusable CUDA
libraries into a quota-limited home directory. On a GPU cluster you want a
CUDA build matched to the site's driver — consult your local documentation
rather than copying step 4 verbatim.

---

## Job script preamble

Every job needs the same clean start as the install did. Without `module
purge`, the default MPI and compiler modules can collide with the libraries
bundled inside torch at runtime.

```bash
#!/bin/bash -l
#$ -N vasp-mace
#$ -l h_rt=2:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -cwd

module purge
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vasp_mace_env

export MACE_MODEL_PATH=$HOME/Scratch/2024-01-07-mace-128-L2_epoch-199.model
export OMP_NUM_THREADS=$NSLOTS      # match the cores actually requested

vasp-mace
```

The `-cwd` flag runs the job in the directory you submitted from, so submit
from the folder holding your `INCAR` and `POSCAR`.

On Slurm, replace the `#$` block with the `#SBATCH` equivalent and use
`$SLURM_CPUS_PER_TASK` in place of `$NSLOTS`; the `module purge` / activate /
`MACE_MODEL_PATH` lines are unchanged.

Leaving `OMP_NUM_THREADS` unset lets torch spawn one thread per physical core
on the node, including cores allocated to other users' jobs — it is worth
setting explicitly.
