"""
SupraMolecule
=============

#. :class:`.SupraMolecule`

SupraMolecule class for optimisation.

"""

import networkx as nx
import numpy as np

from .molecule import Molecule
from .atom import Atom
from .bond import Bond


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

    def with_position_matrix(self, position_matrix):
        """
        Return clone SupraMolecule with new position matrix.

        Parameters
        ----------
        position_matrix : :class:`numpy.ndarray`
            A position matrix of the clone. The shape of the matrix
            is ``(n, 3)``.

        """

        _temp_components = tuple(self.get_components())

        _temp_supramolecule = SupraMolecule(
            atoms=self._atoms,
            bonds=self._bonds,
            position_matrix=np.array(position_matrix),
            cid=self._cid,
            potential=self._potential,
        )
        # Overwrite redefined components.
        _temp_supramolecule._components = _temp_components
        return _temp_supramolecule

    @classmethod
    def init_from_components(
        cls,
        components,
        cid=None,
        potential=None,
    ):
        """
        Initialize a :class:`Supramolecule` instance from components.

        Parameters
        ----------
        components : :class:`iterable` of :class:`.Molecule`
            Molecular components that define the supramolecule.

        cid : :class:`int`, optional
            Conformer id of supramolecule.

        potential : :class:`float`, optional
            Potential energy of Supramolecule.

        """

        atoms = []
        bonds = []
        position_matrix = []
        # Map old atom ids in components to atom ids in supramolecule.
        atom_id_map = {}
        bond_id_map = {}
        for comp in components:
            for a in comp.get_atoms():
                if len(atom_id_map) == 0:
                    atom_id_map[a.get_id()] = 0
                else:
                    atom_id_map[a.get_id()] = max(
                        [i for i in atom_id_map.values()]
                    )+1
                atoms.append(Atom(
                    id=atom_id_map[a.get_id()],
                    element_string=a.get_element_string(),
                ))
            for b in comp.get_bonds():
                if len(bond_id_map) == 0:
                    bond_id_map[b.get_id()] = 0
                else:
                    bond_id_map[b.get_id()] = max(
                        [i for i in bond_id_map.values()]
                    )+1
                bonds.append(Bond(
                    id=bond_id_map[b.get_id()],
                    atom_ids=(
                        atom_id_map[b.get_atom1_id()],
                        atom_id_map[b.get_atom2_id()],
                    )
                ))
            for pos in comp.get_position_matrix():
                position_matrix.append(pos)

        supramolecule = cls.__new__(cls)
        supramolecule._atoms = tuple(atoms)
        supramolecule._bonds = tuple(bonds)
        supramolecule._components = tuple(components)
        supramolecule._cid = cid
        supramolecule._potential = potential
        supramolecule._position_matrix = np.array(position_matrix).T
        return supramolecule

    def _define_components(self):
        """
        Define disconnected component molecules as :class:`.Molecule`s.

        """

        # Produce a graph from the molecule that does not include edges
        # where the bonds to be optimized are.
        mol_graph = nx.Graph()
        for atom in self.get_atoms():
            mol_graph.add_node(atom.get_id())

        # Add edges.
        for bond in self._bonds:
            pair_ids = (bond.get_atom1_id(), bond.get_atom2_id())
            mol_graph.add_edge(*pair_ids)

        # Get atom ids in disconnected subgraphs.
        comps = []
        for c in nx.connected_components(mol_graph):
            c_ids = sorted(c)
            in_atoms = [
                i for i in self._atoms
                if i.get_id() in c
            ]
            in_bonds = [
                i for i in self._bonds
                if i.get_atom1_id() in c and i.get_atom2_id() in c
            ]
            new_pos_matrix = self._position_matrix[:, list(c_ids)].T
            comps.append(
                Molecule(in_atoms, in_bonds, new_pos_matrix)
            )

        self._components = tuple(comps)

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
        content[0] = f'{i}\ncid:{self._cid}, pot: {self._potential}\n'

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
        comps = ', '.join([str(i) for i in self.get_components()])
        return (
            f'{self.__class__.__name__}('
            f'{len(list(self.get_components()))} components, '
            f'{comps})'
        )
