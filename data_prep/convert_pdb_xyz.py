import numpy as np
import argparse
import time
# import tqdm

from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
from ase.io import read, write
from ase.build import molecule
from ase import Atoms
from ase.calculators.cp2k import CP2K
import os

from ase.io import read, write

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default="./")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    # Read the PDB file
    pdb_file = os.path.join(args.savedir, f"output{args.suffix}.pdb")
    xyz_file = os.path.join(args.savedir, f"output{args.suffix}.xyz")
    traj = read(pdb_file, index=':')
    print(f"Number of frames: {len(traj)}")
    for i, atoms in enumerate(traj):
        if i == 0:
            cell = atoms.cell  # Use the cell of the first frame
        atoms.set_cell(cell)
        atoms.set_pbc([1, 1, 1])

    write(xyz_file, traj)
