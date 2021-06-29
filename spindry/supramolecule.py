"""
SupraMolecule
=============

#. :class:`.SupraMolecule`

SupraMolecule class for optimisation.

"""

import networkx as nx
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


    def _define_components(self):
        """
        Define disconnected component molecules as :class:`.Molecule`s.

        """

        # Produce a graph from the molecule that does not include edges
        # where the bonds to be optimized are.
        mol_graph = nx.Graph()
        # Add edges.
        for bond in self._bonds:
            pair_ids = (bond.get_atom1_id(), bond.get_atom2_id())
            mol_graph.add_edge(*pair_ids)

        # Get atom ids in disconnected subgraphs.
        comps = []
        for c in nx.connected_components(mol_graph):
            in_atoms = [
                i for i in self._atoms
                if i.get_id() in c
            ]
            in_bonds = [
                i for i in self._bonds
                if i.get_atom1_id() in c and i.get_atom2_id() in c
            ]
            new_pos_matrix = self._position_matrix[:, list(c)].T
            comps.append(
                Molecule(in_atoms, in_bonds, new_pos_matrix)
            )

        self._components = tuple(comps)

        # Set first line to the atom_count.
        content[0] = f'{i+j}\ncid:{self._cid}, pot:{self._potential}\n'

        return content

    def get_components(self):
        """
        Yields each molecular component.

        """

        for i in self._components:
            yield i

    def get_cid(self):
        return self._cid

    def get_potential(self):
        return self._potential

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(list(self.get_components()))} components, '
            f'{comps})'
        )
