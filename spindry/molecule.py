"""
Molecule
========

#. :class:`.Molecule`

Molecule class for optimisation.

"""

import numpy as np

from .atom import Atom


class Molecule:
    """
    Representation of a molecule containing atoms and positions.

    """

    def __init__(self, path):
        """
        Initialize a :class:`Molecule` instance.

        Parameters
        ----------
        path : :class:`str`
            Path to `.xyz` file defining molecule to read.

        Raises
        ------
        :class:`RuntimeError`
            If the number of atoms in the content of the file does not
            match the number of atoms at the top of the file.

        """

        # Load in xyz file.
        with open(path, 'r') as f:
            atom_count, _, *content = f.readlines()

        # Save all the coords in the file.
        new_coords = []
        atoms = []
        for i, line in enumerate(content):
            element, *coords = line.split()
            # Handle XYZ files with capitilisation of element symbols.
            element = element.title()
            atoms.append(Atom(id=i, element_string=str(element)))
            new_coords.append([float(i) for i in coords])

        # Check that the correct number of atom
        # lines was present in the file.
        if i+1 != int(atom_count):
            raise RuntimeError(
                f'The number of atom lines in the xyz file, {i+1}, '
                'does not match the number of atoms in the '
                f'content, {atom_count}.'
            )

        new_coords = np.array(new_coords)

        self._atoms = tuple(atoms)
        self._position_matrix = np.array(
            new_coords.T,
            dtype=np.float64,
        )

    def init(self, atoms, position_matrix):
        """
        Init a Molecule with from atoms and position matrix.

        Parameters
        ----------
        atoms : :class:`iterable` of :class:`.Atom`
            Atoms that define the molecule.

        position_matrix : :class:`numpy.ndarray`
            A position matrix of the clone. The shape of the matrix
            is ``(n, 3)``.

        """
        self._atoms = tuple(atoms)
        self._position_matrix = np.array(position_matrix).T

    def get_position_matrix(self):
        """
        Return a matrix holding the atomic positions.

        Returns
        -------
        :class:`numpy.ndarray`
            The array has the shape ``(n, 3)``. Each row holds the
            x, y and z coordinates of an atom.

        """

        return np.array(self._position_matrix.T)

    def with_centroid(self, position):
        """
        Return clone Molecule at position.

        Parameters
        ----------
        position : :class:`numpy.ndarray`
            The position of the centroid. The shape of the matrix
            is ``(3, )``.

        """

        centroid = self.get_centroid()
        return self.with_displacement(position-centroid)

    def with_displacement(self, displacement):
        """
        Return a displaced clone Molecule.

        Parameters
        ----------
        displacement : :class:`numpy.ndarray`
            The displacement vector to be applied.

        """

        new_position_matrix = (
            self._position_matrix.T + displacement
        )

        clone = self.__class__.__new__(self.__class__)
        Molecule.init(
            self=clone,
            atoms=self._atoms,
            position_matrix=np.array(new_position_matrix),
        )
        return clone

    def with_position_matrix(self, position_matrix):
        """
        Return clone Molecule with new position matrix.

        Parameters
        ----------
        position_matrix : :class:`numpy.ndarray`
            A position matrix of the clone. The shape of the matrix
            is ``(n, 3)``.

        """

        clone = self.__class__.__new__(self.__class__)
        Molecule.init(
            self=clone,
            atoms=self._atoms,
            position_matrix=np.array(position_matrix),
        )
        return clone

    def _write_xyz_content(self):
        """
        Write basic `.xyz` file content of Molecule.

        """
        coords = self.get_position_matrix()
        content = [0]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )
        # Set first line to the atom_count.
        content[0] = f'{i}\n\n'

        return content

    def write_xyz_file(self, path):
        """
        Write basic `.xyz` file of Molecule to `path`.

        Connectivity is not maintained in this file type!

        """

        content = self._write_xyz_content()

        with open(path, 'w') as f:
            f.write(''.join(content))

    def get_atoms(self):
        """
        Yield the atoms in the molecule, ordered as input.

        Yields
        ------
        :class:`.Atom`
            An atom in the molecule.

        """

        for atom in self._atoms:
            yield atom

    def get_num_atoms(self):
        """
        Return the number of atoms in the molecule.

        """

        return len(self._atoms)

    def get_centroid(self, atom_ids=None):
        """
        Return the centroid.

        Parameters
        ----------
        atom_ids : :class:`iterable` of :class:`int`, optional
            The ids of atoms which are used to calculate the
            centroid. Can be a single :class:`int`, if a single
            atom is to be used, or ``None`` if all atoms are to be
            used.

        Returns
        -------
        :class:`numpy.ndarray`
            The centroid of atoms specified by `atom_ids`.

        Raises
        ------
        :class:`ValueError`
            If `atom_ids` has a length of ``0``.

        """

        if atom_ids is None:
            atom_ids = range(len(self._atoms))
        elif isinstance(atom_ids, int):
            atom_ids = (atom_ids, )
        elif not isinstance(atom_ids, (list, tuple)):
            atom_ids = list(atom_ids)

        if len(atom_ids) == 0:
            raise ValueError('atom_ids was of length 0.')

        return np.divide(
            self._position_matrix[:, atom_ids].sum(axis=1),
            len(atom_ids)
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}({len(self._atoms)} atoms) '
            f'at {id(self)}>'
        )
