"""
SupraMolecule
=============

#. :class:`.SupraMolecule`

SupraMolecule class for optimisation.

"""

import numpy as np

from .molecule import Molecule


class SupraMolecule(Molecule):
    """
    Representation of a supramolecule containing atoms and positions.

    """

    def __init__(self, host, guest, cid, potential):
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

        self._host = host
        self._guest = guest
        self._cid = cid
        self._potential = potential

    def _write_xyz_content(self):
        host_coords = self._host.get_position_matrix()
        guest_coords = self._guest.get_position_matrix()
        content = [0]

        for i, atom in enumerate(self._host.get_atoms(), 1):
            x, y, z = (i for i in host_coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )

        for j, atom in enumerate(self._guest.get_atoms(), 1):
            x, y, z = (i for i in guest_coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )

        # Set first line to the atom_count.
        content[0] = f'{i+j}\n\n'

        return content

    def get_host(self):
        return self._host

    def get_guest(self):
        return self._guest

    def get_cid(self):
        return self._cid

    def get_potential(self):
        return self._potential

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}('
            f'{self._host.get_num_atoms()} + '
            f'{self._guest.get_num_atoms()} atoms) '
            f'at {id(self)}>'
        )
