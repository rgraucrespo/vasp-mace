# vasp-mace

**VASP-like interface for structure relaxation and energy calculations using MACE machine-learning potentials**

`vasp-mace` emulates the behaviour of VASP for quick, low-cost structure optimisations using pretrained MACE interatomic potentials, optionally including empirical dispersion corrections.  
It reads VASP-style inputs (`POSCAR`, `INCAR`) and produces outputs (`CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`), allowing seamless integration with existing VASP workflows.

--- 

## 1. Installation

git clone https://github.com/rgraucrespo/vasp-mace.git
cd vasp-mace
conda create -n vasp_mace_env python=3.11 -y
conda activate vasp_mace_env
pip install ase torch mace-torch
conda install -c conda-forge dftd4
pip install -e .

--- 

## 2. Model setup

wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model

Specify this path in your vasp_mace.yaml file.

--- 

## 3. Usage

Prepare POSCAR and INCAR as in VASP, then run:
vasp-mace

This parses inputs, loads the MACE model, performs relaxation, and writes outputs like VASP.

--- 

## 4. Example

INCAR:
NSW = 99
ISIF = 3
EDIFFG = -0.01
PSTRESS = 20

vasp_mace.yaml:
model: /home/user/software/mace/2024-01-07-mace-128-L2_epoch-199.model
device: cuda
dtype: float64
dispersion: false

--- 

## 5. License and citation

MIT License © 2025 Ricardo Grau-Crespo.

Cite:
- Batatia et al., *Nat. Commun.* 14, 5083 (2023)
- Grau-Crespo et al., *unpublished tools and workflows*, 2025
