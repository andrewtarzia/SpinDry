"""Generator of host guest conformations using nonbonded interactions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from .potential import SpdPotential
from .supramolecule import SupraMolecule
from .utilities import rotation_matrix_arbitrary_axis

if TYPE_CHECKING:
    from collections import abc

    import mchammer as mch


class Spinner:
    """Generate host-guest conformations by rotating guest.

    A Metroplis MC algorithm is applied to perform rigid
    translations and rotations of the guest, relative to the host.

    """

    def __init__(  # noqa: PLR0913
        self,
        step_size: float,
        rotation_step_size: float,
        num_conformers: int,
        max_attempts: int = 1000,
        potential_function: SpdPotential | None = None,
        beta: float = 2,
        random_seed: int | None = 1000,
    ) -> None:
        """Initialize a :class:`Spinner` instance.

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

        potential_function : :class:`spd.Potential`
            Function to calculate potential energy of a
            :class:`spd.Supramolecule`

        beta : :class:`float`, optional
            Value of beta used in the in MC moves. Beta takes the
            place of the inverse boltzmann temperature.
            Defaults to 2.

        random_seed : :class:`int` or :class:`NoneType`, optional
            Random seed to use for MC algorithm. Should only be set to
            ``None`` if system-based random seed is desired. Defaults
            to a set seed of 1000, to avoid randomness.

        """
        self._step_size = step_size
        self._num_conformers = num_conformers
        self._rotation_step_size = rotation_step_size
        self._max_attempts = max_attempts
        if potential_function is None:
            self._potential_function = SpdPotential(5)
        else:
            self._potential_function = potential_function
        self._beta = beta
        if random_seed is None:
            self._generator = np.random.default_rng()
        else:
            self._generator = np.random.default_rng(random_seed)

    def compute_potential(self, supramolecule: SupraMolecule) -> float:
        return self._potential_function.compute_potential(supramolecule)

    def _translate_atoms_along_vector(
        self,
        mol: SupraMolecule,
        vector: np.ndarray,
    ) -> SupraMolecule:
        return mol.with_displacement(vector)

    def _rotate_atoms_by_angle(
        self,
        mol: SupraMolecule,
        angle: float,
        axis: np.ndarray,
        origin: np.ndarray,
    ) -> SupraMolecule:
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

        return mol.with_position_matrix(new_position_matrix)

    def _test_move(self, curr_pot: float, new_pot: float) -> bool:
        if new_pot < curr_pot:
            return True

        exp_term = np.exp(-self._beta * (new_pot - curr_pot))
        rand_number = random.random()  # noqa: S311

        return exp_term > rand_number

    def _run_step(
        self,
        supramolecule: SupraMolecule,
        movable_components: tuple[int, ...] | None,
    ) -> tuple[SupraMolecule, float]:
        component_list = list(supramolecule.get_components())
        component_sizes = {
            i: mol.get_num_atoms() for i, mol in enumerate(component_list)
        }
        max_size = max(component_sizes.values())

        # If movable components not set, select a guest randomly to
        # move and reorient.
        # Do not move or rotate largest component.
        if movable_components is None:
            # If there are different sizes.
            if len(set(component_sizes.values())) > 1:
                movable_components = tuple(
                    i
                    for i in range(len(component_list))
                    if component_sizes[i] != max_size
                )
            # Else capture all!
            else:
                movable_components = tuple(
                    i for i in range(len(component_list))
                )

        targ_comp_id = self._generator.choice(
            [i for i in range(len(component_list)) if i in movable_components]
        )

        targ_comp = component_list[targ_comp_id]

        # Random number from -1 to 1 for multiplying translation.
        rand = (self._generator.random() - 0.5) * 2

        # Random translation direction.
        rand_vector = self._generator.random(3)
        rand_vector = rand_vector / np.linalg.norm(rand_vector)

        # Perform translation.
        translation_vector = rand_vector * self._step_size * rand
        targ_comp = mch.translate_molecule_along_vector(
            mol=targ_comp,
            vector=translation_vector,
        )

        # Define a random rotation of the guest.
        # Random number from -1 to 1 for multiplying rotation.
        rand = (self._generator.random() - 0.5) * 2
        rotation_angle = self._rotation_step_size * rand
        rand_axis = self._generator.random(3)
        rand_axis = rand_axis / np.linalg.norm(rand_vector)

        # Perform rotation.
        targ_comp = self._rotate_atoms_by_angle(
            mol=targ_comp,
            angle=rotation_angle,
            axis=rand_axis,
            origin=targ_comp.get_centroid(),
        )

        component_list[targ_comp_id] = targ_comp
        supramolecule = SupraMolecule.init_from_components(
            components=component_list,
        )

        nonbonded_potential = self._compute_potential(supramolecule)
        return supramolecule, nonbonded_potential

    def get_conformers(
        self,
        supramolecule: SupraMolecule,
        movable_components: tuple[int, ...] | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
    ) -> abc.Iterable[mch.Molecule]:
        """Get conformers of supramolecule.

        Parameters
        ----------
        supramolecule : :class:`.SupraMolecule`
            The supramolecule to optimize.

        movable_components : :class:`iterable` of :class:`int`,
        optional
            Components of supramolecule to move during simulation.

        verbose : :class:`bool`
            `True` to print some extra information.

        Yields:
        ------
        conformer : :class:`.SupraMolecule`
            The host-guest supramolecule.

        """
        cid = 0
        nonbonded_potential = self._compute_potential(supramolecule)

        yield SupraMolecule.init_from_components(
            components=list(supramolecule.get_components()),
            cid=cid,
            potential=nonbonded_potential,
        )
        cids_passed = []
        count = 0
        for _ in range(1, self._max_attempts):
            n_supramolecule, n_nonbonded_potential = self._run_step(
                supramolecule=supramolecule,
                movable_components=movable_components,
            )
                new_pot=n_nonbonded_potential,
                generator=self._generator,
            )
            if passed:
                cid += 1
                cids_passed.append(cid)
                nonbonded_potential = n_nonbonded_potential
                supramolecule = SupraMolecule.init_from_components(
                    components=list(n_supramolecule.get_components()),
                    cid=cid,
                    potential=nonbonded_potential,
                )
                supramolecule = supramolecule.with_position_matrix(
                    n_supramolecule.get_position_matrix()
                )
                yield supramolecule
            count += 1
            if len(cids_passed) == self._num_conformers:
                break

        if verbose:
            print(  # noqa: T201
                f"{len(cids_passed)} conformers generated in {count} steps."
            )

    def get_final_conformer(
        self,
        supramolecule: SupraMolecule,
        movable_components: tuple[int, ...] | None = None,
    ) -> mch.Molecule:
        """Get final conformer of supramolecule.

        Parameters
        ----------
        supramolecule : :class:`.SupraMolecule`
            The supramolecule to optimize.

        movable_components : :class:`iterable` of :class:`int`,
        optional
            Components of supramolecule to move during simulation.
            If `None`, then moved components are selected randomly,
            and the largest component (host) is not moved.

        Returns:
        -------
        conformer : :class:`.SupraMolecule`
            The host-guest supramolecule.

        """
        for conformer in self.get_conformers(  # noqa: B007
            supramolecule=supramolecule,
            movable_components=movable_components,
        ):
            continue

        return conformer
