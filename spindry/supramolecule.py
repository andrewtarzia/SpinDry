"""
SupraMolecule
=============

#. :class:`.SupraMolecule`

SupraMolecule class for optimisation.

"""

from .molecule import Molecule


class SupraMolecule(Molecule):
    """
    Representation of a supramolecule containing atoms and positions.

    """

    def __init__(self, host, guests, cid, potential):
        """
        Initialize a :class:`Molecule` instance.

        Parameters
        ----------
        host : :class:`.Molecule`
            The host molecule.

        guests : :class:`list` of :class:`Molecule`
            The guest molecules.

        cid : :class:`int`
            Conformer id of supramolecule.

        potential : :class:`float`
            Potential energy of Supramolecule.

        Raises
        ------
        :class:`RuntimeError`
            If the number of atoms in the content of the file does not
            match the number of atoms at the top of the file.

        """

        self._host = host
        self._guests = guests
        self._cid = cid
        self._potential = potential

    def _write_xyz_content(self):
        content = [0]

        host_coords = self._host.get_position_matrix()
        for i, atom in enumerate(self._host.get_atoms(), 1):
            x, y, z = (i for i in host_coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )

        for guest in self._guests:
            guest_coords = guest.get_position_matrix()
            for j, atom in enumerate(guest.get_atoms(), 1):
                x, y, z = (i for i in guest_coords[atom.get_id()])
                content.append(
                    f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
                )

        # Set first line to the atom_count.
        content[0] = f'{i+j}\ncid:{self._cid}, pot:{self._potential}\n'

        return content

    def get_host(self):
        return self._host

    def get_guests(self):
        return self._guests

    def get_cid(self):
        return self._cid

    def get_potential(self):
        return self._potential

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (f'<{self.__class__.__name__} at {id(self)}>')
