import mchammer as mch  # noqa: INP001, D100
import spindry as spd
import stk

# Building a cage from the examples on the stk docs.
bb1 = stk.BuildingBlock(
    smiles="O=CC(C=O)C=O",
    functional_groups=[stk.AldehydeFactory()],
)
bb2 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])

cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        building_blocks=(bb1, bb2),
    ),
)
stk_guests = (stk.BuildingBlock("C1CCCCC1"),)

host_guest = stk.ConstructedMolecule(
    topology_graph=stk.host_guest.Complex(
        host=stk.BuildingBlock.init_from_molecule(cage),
        guests=[stk.host_guest.Guest(i) for i in stk_guests],
    )
)

host_molecule = mch.Molecule(
    atoms=(
        mch.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        )
        for atom in cage.get_atoms()
    ),
    bonds=(
        mch.Bond(
            id=i,
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            ),
        )
        for i, bond in enumerate(cage.get_bonds())
    ),
    position_matrix=(cage.get_position_matrix()),
)
guest_molecule = mch.Molecule(
    atoms=(
        mch.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        )
        for atom in stk_guests[0].get_atoms()
    ),
    bonds=(
        mch.Bond(
            id=i,
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            ),
        )
        for i, bond in enumerate(stk_guests[0].get_bonds())
    ),
    position_matrix=stk_guests[0].get_position_matrix(),
)

supramolecule = spd.SupraMolecule.init_from_components(
    components=(host_molecule, guest_molecule),
)
cg = spd.Spinner(
    step_size=0.2,
    rotation_step_size=1.0,
    num_conformers=5,
)

for conformer in cg.get_conformers(supramolecule):  # noqa: B007
    pass

# Write optimised structures out.
for i, comp in enumerate(list(conformer.get_components())):
    if i == 0:
        new_cage = cage.with_position_matrix(comp.get_position_matrix())
        new_cage.write("comp_new_host.mol")
    elif i == 1:
        new_guest = stk_guests[0].with_position_matrix(
            comp.get_position_matrix()
        )
        new_guest.write("comp_new_guest.mol")
