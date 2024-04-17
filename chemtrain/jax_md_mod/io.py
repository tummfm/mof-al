"""Functions for io: Loading data to and from Jax M.D."""

import jax.numpy as jnp
import mdtraj
import numpy as np


def load_box(filename):
    """Loads initial configuration using the file loader from MDTraj.

    Args:
        filename: String providing the location of the file to load.

    Returns:
        Tuple of jnp arrays of box, coordinates, species and mass.
    """
    traj = mdtraj.load(filename)
    coordinates = traj.xyz[0]
    box = traj.unitcell_lengths[0]

    species = np.zeros(coordinates.shape[0])
    masses = np.zeros_like(species)
    for atom in traj.topology.atoms:
        species[atom.index] = atom.element.number
        masses[atom.index] = atom.element.mass

    return (jnp.array(box), jnp.array(coordinates), jnp.array(masses),
            jnp.array(species, dtype=jnp.int32))