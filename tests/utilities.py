import itertools as it

import mchammer as mch


def is_equivalent_spd_atom(atom1: mch.Atom, atom2: mch.Atom) -> None:
    """Test if atoms are equivalent."""
    assert atom1.get_id() == atom2.get_id()
    assert atom1.get_element_string() == atom2.get_element_string()
    assert atom1.get_radius() == atom2.get_radius()
    assert atom1.__class__ is atom2.__class__


def is_equivalent_spd_bond(bond1: mch.Bond, bond2: mch.Bond) -> None:
    """Test if bonds are equivalent."""
    assert bond1.__class__ is bond2.__class__
    assert bond1.get_id() == bond2.get_id()
    assert bond1.get_atom1_id() == bond2.get_atom1_id()
    assert bond1.get_atom2_id() == bond2.get_atom2_id()


def is_equivalent_spd_molecule(
    molecule1: mch.Molecule,
    molecule2: mch.Molecule,
) -> None:
    """Test if molecules are equivalent."""
    print(molecule1, molecule2)
    atoms = it.zip_longest(
        molecule1.get_atoms(),
        molecule2.get_atoms(),
    )
    for atom1, atom2 in atoms:
        is_equivalent_spd_atom(atom1, atom2)

    bonds = it.zip_longest(
        molecule1.get_bonds(),
        molecule2.get_bonds(),
    )
    for bond1, bond2 in bonds:
        is_equivalent_spd_bond(bond1, bond2)
