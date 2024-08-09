"""Run Steered Molecular Dynamics with ASE and PySAGES for data creation.
"""
import argparse
import os
import shutil
from functools import partial
from pysages.backends import SamplingContext
from steered import Steered, SteeredLogger, DistanceToInterface, Orientational_Order_Parameter_QTensor, Orientational_Order_Parameter_Axis, CVLogger

import openmm
import openmm.app as app
import openmm.unit as unit
import jax.numpy as np
import numpy
import pysages
from pysages.methods import Unbiased, HistogramLogger

pi = numpy.pi
kB = 0.008314462618

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initdir", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--offsetstyle", type=int, default=0)
    parser.add_argument("--firstrun", action="store_true")
    parser.add_argument("--cvstyle", type=str, default="qtensor",
                        choices=["qtensor", "axis"])
    parser.add_argument("--unbiased", action="store_true")
    return parser.parse_args()


def generate_simulation(T=1000.0 * unit.kelvin,
                        dt=1.0 * unit.femtoseconds,
                        initdir: str="",  # Directory that contains initial config
                        savedir: str="",  # Directory that saves the results
                        first_run: bool=False,
                        unbiased: bool=False):
    print("Loading AMBER files...")
    prmtop = app.AmberPrmtopFile(
        os.path.join(initdir, "5cb.prmtop")
    )
    # If it is the first run, starts from the output_nvt.pdb.
    if first_run or unbiased:
        pdb = app.PDBFile(os.path.join(initdir, "output_nvt.pdb"))
        shutil.copyfile(os.path.join(initdir, "5cb.prmtop"),
                        os.path.join(savedir, "5cb.prmtop"))
    else:
        pdb = app.PDBFile(os.path.join(initdir, "output_init.pdb"))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        rigidWater=True,
        switchDistance=0.6 * unit.nanometer,
        nonbondedCutoff=0.8 * unit.nanometer,
        constraints=None,
    )
    system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Create the integrator to do Langevin dynamics
    integrator = openmm.LangevinIntegrator(
        T,  # Temperature of heat bath
        1.0 / unit.picoseconds,  # Friction coefficient
        dt,  # Time step
    )

    # Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
    platform = openmm.Platform.getPlatformByName("CUDA")

    # Create the Simulation object
    sim = app.Simulation(pdb.topology, system, integrator, platform)

    # Set the particle positions
    sim.context.setPositions(pdb.getPositions(frame=-1))
    if first_run:
        pdb_output = os.path.join(savedir, "output_init.pdb")
        data_output = os.path.join(savedir, "data_init.txt")
        pdb_report_interval = 10000
    else:
        pdb_output = os.path.join(savedir, "output.pdb")
        data_output = os.path.join(savedir, "data.txt")
        pdb_report_interval = 2000
    sim.reporters.append(app.PDBReporter(pdb_output, pdb_report_interval))
    sim.reporters.append(
        app.StateDataReporter(
            data_output,
            10000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            totalSteps=1e6,
            append=True
        )
    )

    return sim


def get_indices(offset_style=0,
                num_molecules=18):
    atom_indices = [] # indices
    pair_indices = [] # sequences
    if offset_style == 0:
        # Second Ring Carbon: 5
        # First Ring Carbon: 11
        # Nitrogen: 18
        offset_nitrogen = 18
        offset_carbon = 11
    elif offset_style == 1:
        # Nitrogen is used as the beginning
        offset_nitrogen = 0
        offset_carbon = 5
    for i in range(0, num_molecules):
        atom_index = [i * 38 + offset_nitrogen,
                      i * 38 + offset_carbon]
        pair_index = [2 * i, 2 * i + 1]
        atom_indices.append(atom_index)
        pair_indices.append(pair_index)
    return atom_indices, pair_indices


def main(args):
    atom_indices, pair_indices = get_indices(args.offsetstyle)
    first_run = args.firstrun
    savedir = args.savedir
    initdir = args.initdir
    os.makedirs(savedir, exist_ok=True)

    print("Warning: Backend is OpenMM, please notice that the unit is different.")

    if args.cvstyle == "qtensor":
        cvs = (Orientational_Order_Parameter_QTensor(atom_indices, 2, np.array(pair_indices)),)
    elif args.cvstyle == "axis":
        cvs = (Orientational_Order_Parameter_Axis(atom_indices, 2, np.array(pair_indices)),)
    else:
        raise ValueError(f"Unsupported CV Style: {args.cvstyle}.")

    k = 10000
    if first_run:
        center_cv = np.array([0.5])
        steer_velocity = np.array([-1 / 1e3])  # Setting velocity to -0.5 for creating init
        logger_interval = 10000
        steered_file = os.path.join(savedir, "steered_init.dat")
        timesteps = int(1e6)
    else:
        center_cv = np.array([-0.5])
        steer_velocity = np.array([0.00075])
        logger_interval = 2000
        steered_file = os.path.join(savedir, "steered.dat")
        timesteps = int(2e6)
    if not args.unbiased:
        method = Steered(cvs, k, center_cv, steer_velocity)
        callback = SteeredLogger(steered_file, logger_interval)
    else:
        method = Unbiased(cvs)
        callback = CVLogger(steered_file, logger_interval)
    generate_simulation_partial = partial(generate_simulation,
                                          initdir=initdir,
                                          savedir=savedir,
                                          first_run=first_run,
                                          unbiased=args.unbiased)
    sampling_context = SamplingContext(method, generate_simulation_partial, callback)
    # Run the steered dynamics
    pysages.run(sampling_context, timesteps)


if __name__ == "__main__":
    args = get_args()
    main(args)
