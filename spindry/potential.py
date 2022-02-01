"""
Potential
=========

#. :class:`.Potential`

Classes for calculating the potential energy of supramolecules.

"""

import numpy as np

from itertools import combinations
from scipy.spatial.distance import cdist


class Potential:
    """
    Base class for potential calculators.

    """

    def __init__(self):
        """
        Initialize a :class:`Spinner` instance.

        """

    def compute_potential(self, supramolecule):
        """
        Calculate potential energy.

        Parameters
        ----------
        supramolecular : :class:`spd.Supramolecule`
            Supramolecule to evaluate.

        """
        raise NotImplementedError()



class SpdPotential:
    """
    Default spindry non-bonded potential function.

    """

    def __init__(self, nonbond_epsilon=5):
        """
        Initialize a :class:`Spinner` instance.

        Parameters
        ----------
        nonbond_epsilon : :class:`float`, optional
            Value of epsilon used in the nonbond potential in MC moves.
            Determines strength of the nonbond potential.
            Defaults to 20.

        """

        self._nonbond_epsilon = nonbond_epsilon

    def _nonbond_potential(self, distance, sigmas):
        """
        Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (sigmas/distance) ** 12 - (sigmas/distance) ** 6
            )
        )

    def _mixing_function(self, val1, val2):
        return (val1 + val2) / 2

    def _combine_sigma(self, radii1, radii2):
        """
        Combine radii using Lorentz-Berthelot rules.

        """

        len1 = len(radii1)
        len2 = len(radii2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = self._mixing_function(
                    radii1[i], radii2[j],
                )

        return mixed

    def _compute_nonbonded_potential(self, position_matrices, radii):
        nonbonded_potential = 0
        for pos_mat_pair, radii_pair in zip(
            combinations(position_matrices, 2),
            combinations(radii, 2),
        ):
            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            sigmas =  self._combine_sigma(radii_pair[0], radii_pair[1])
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigmas=sigmas.flatten(),
                )
            )

        return nonbonded_potential

    def compute_potential(self, supramolecule):
        component_position_matrices = (
            i.get_position_matrix()
            for i in supramolecule.get_components()
        )
        component_radii = (
            tuple(j.get_radius() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        return self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
        )
