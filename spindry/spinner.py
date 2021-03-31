"""
Spinner
=======

#. :class:`.Spinner`

Generator of host guest conformations using nonbonded interactions.

"""

import numpy as np

from scipy.spatial.distance import cdist
from copy import deepcopy
import random

from .supramolecule import SupraMolecule
from .utilities import rotation_matrix_arbitrary_axis


class Spinner:
    """
    Generate host-guest conformations by rotating guest.

    A Metroplis MC algorithm is applied to perform rigid
    translations and rotations of the guest, relative to the host.

    """

    def __init__(
        self,
        step_size,
        rotation_step_size,
        num_conformers,
        max_attempts=1000,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
        beta=2,
        random_seed=1000,
    ):
        """
        Initialize a :class:`Spinner` instance.

        Parameters
        ----------
        step_size : :class:`float`
            The relative size of the step to take during step.

        rotation_step_size : :class:`float`
            The relative size of the rotation to take during step.

        num_conformers : :class:`int`
            Number of conformers to extract.

        max_attempts : :class:`int`
            Maximum number of MC moves to try to generate conformers.

        nonbond_epsilon : :class:`float`, optional
            Value of epsilon used in the nonbond potential in MC moves.
            Determines strength of the nonbond potential.
            Defaults to 20.

        nonbond_sigma : :class:`float`, optional
            Value of sigma used in the nonbond potential in MC moves.
            Defaults to 1.2.

        nonbond_mu : :class:`float`, optional
            Value of mu used in the nonbond potential in MC moves.
            Determines the steepness of the nonbond potential.
            Defaults to 3.

        beta : :class:`float`, optional
            Value of beta used in the in MC moves. Beta takes the
            place of the inverse boltzmann temperature.
            Defaults to 2.

        random_seed : :class:`int` or :class:`NoneType`, optional
            Random seed to use for MC algorithm. Should only be set to
            ``None`` if system-based random seed is desired. Defaults
            to 1000.

        """

        self._step_size = step_size
        self._num_conformers = num_conformers
        self._rotation_step_size = rotation_step_size
        self._max_attempts = max_attempts
        self._nonbond_epsilon = nonbond_epsilon
        self._nonbond_sigma = nonbond_sigma
        self._nonbond_mu = nonbond_mu
        self._beta = beta
        if random_seed is None:
            np.random.seed()
            random.seed()
        else:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def _nonbond_potential(self, distance):
        """
        Define an arbitrary repulsive nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (self._nonbond_sigma/distance) ** self._nonbond_mu
            )
        )

    def _compute_nonbonded_potential(
        self,
        host_position_matrix,
        guest_position_matrix
    ):
        # Get all pairwise distances between atoms in host and guest.
        pair_dists = cdist(
            host_position_matrix, guest_position_matrix
        )
        nonbonded_potential = np.sum(
            self._nonbond_potential(pair_dists.flatten())
        )

        return nonbonded_potential

    def _compute_potential(self, host, guest):
        host_position_matrix = host.get_position_matrix()
        guest_position_matrix = guest.get_position_matrix()
        nonbonded_potential = self._compute_nonbonded_potential(
            host_position_matrix=host_position_matrix,
            guest_position_matrix=guest_position_matrix,
        )

        return nonbonded_potential

    def _translate_atoms_along_vector(self, mol, vector):

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos - vector

        mol = mol.with_position_matrix(new_position_matrix)
        return mol

    def _rotate_atoms_by_angle(self, mol, angle, axis, origin):
        new_position_matrix = mol.get_position_matrix()
        # Set the origin of the rotation to "origin".
        new_position_matrix = new_position_matrix - origin
        # Perform rotation.
        rot_mat = rotation_matrix_arbitrary_axis(angle, axis)
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        new_position_matrix = (rot_mat @ new_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        new_position_matrix = new_position_matrix + origin

        mol = mol.with_position_matrix(new_position_matrix)
        return mol

    def _test_move(self, curr_pot, new_pot):

        if new_pot < curr_pot:
            return True
        else:
            exp_term = np.exp(-self._beta*(new_pot-curr_pot))
            rand_number = random.random()

            if exp_term > rand_number:
                return True
            else:
                return False

    def _run_first_step(self, host, guest):

        host = host.with_centroid([0, 0, 0])
        guest = guest.with_centroid([0, 0, 0])
        nonbonded_potential = self._compute_potential(host, guest)
        return host, guest, nonbonded_potential

    def _run_step(self, host, guest):

        # Random number from -1 to 1 for multiplying translation.
        rand = (random.random() - 0.5) * 2

        # Random translation direction.
        rand_vector = np.random.rand(3)
        rand_vector = rand_vector / np.linalg.norm(rand_vector)

        # Perform translation.
        translation_vector = rand_vector * self._step_size * rand
        guest = self._translate_atoms_along_vector(
            mol=guest,
            vector=translation_vector,
        )

        # Define a random rotation of the guest.
        # Random number from -1 to 1 for multiplying rotation.
        rand = (random.random() - 0.5) * 2
        rotation_angle = self._rotation_step_size * rand
        rand_axis = np.random.rand(3)
        rand_axis = rand_axis / np.linalg.norm(rand_vector)

        # Perform rotation.
        guest = self._rotate_atoms_by_angle(
            mol=guest,
            angle=rotation_angle,
            axis=rand_axis,
            origin=guest.get_centroid(),
        )

        nonbonded_potential = self._compute_potential(host, guest)
        return host, guest, nonbonded_potential

    def get_conformers(self, host, guest):
        """
        Get conformers of guest in host.

        Parameters
        ----------
        host : :class:`.Molecule`
            The host molecule.

        guest : :class:`.Molecule`
            The guest molecule to be manipulated.

        Yields
        ------
        conformer : :class:`.SupraMolecule`
            The host-guest supramolecule.

        """

        cid = 0
        host, guest, nonbonded_potential = self._run_first_step(
            host=host,
            guest=guest,
        )
        yield SupraMolecule(
            host=host,
            guest=guest,
            cid=cid,
            potential=nonbonded_potential,
        )

        cids_passed = [cid]
        for step in range(1, self._max_attempts):
            n_host, n_guest, n_nonbonded_potential = self._run_step(
                host=host,
                guest=guest,
            )
            passed = self._test_move(
                curr_pot=nonbonded_potential,
                new_pot=n_nonbonded_potential
            )
            if passed:
                cid += 1
                yield SupraMolecule(
                    host=host,
                    guest=guest,
                    cid=cid,
                    potential=nonbonded_potential,
                )
                cids_passed.append(cid)
                nonbonded_potential = n_nonbonded_potential
                host = host.with_position_matrix(
                    position_matrix=n_host.get_position_matrix()
                )
                guest = guest.with_position_matrix(
                    position_matrix=n_guest.get_position_matrix()
                )

            if len(cids_passed) == self._num_conformers:
                break

        print(
            f'{len(cids_passed)} conformers generated in {step} steps.'
        )
