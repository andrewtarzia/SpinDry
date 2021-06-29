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

    def __init__(
        self,
        atoms,
        bonds,
        position_matrix,
        cid=None,
        potential=None,
    ):
        """
        Initialize a :class:`Supramolecule` instance.

        Parameters
        ----------
        atoms : :class:`iterable` of :class:`.Atom`
            Atoms that define the molecule.

        bonds : :class:`iterable` of :class:`.Bond`
            Bonds between atoms that define the molecule.

        position_matrix : :class:`numpy.ndarray`
            A ``(n, 3)`` matrix holding the position of every atom in
            the :class:`.Molecule`.

        cid : :class:`int`, optional
            Conformer id of supramolecule.

        potential : :class:`float`, optional
            Potential energy of Supramolecule.

        """

        self._atoms = tuple(atoms)
        self._bonds = tuple(bonds)
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )
        self._define_components()
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
