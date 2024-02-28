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
cage.write("poc.mol")
# Always ensure initial structures are not entirely on top of each
# other.
stk_guests = (
    (stk.BuildingBlock("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), (0.0, 0.0, 0.0)),
    (stk.BuildingBlock("c1ccccc1"), (0.0, 2.0, 0.0)),
)

host_guest = stk.ConstructedMolecule(
    topology_graph=stk.host_guest.Complex(
        host=stk.BuildingBlock.init_from_molecule(cage),
        guests=[
            stk.host_guest.Guest(i[0], displacement=i[1]) for i in stk_guests
        ],
    )
)
host_guest.write("host_multi_guest.mol")

supramolecule = spd.SupraMolecule(
    atoms=(
        mch.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        )
        for atom in host_guest.get_atoms()
    ),
    bonds=(
        mch.Bond(
            id=i,
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            ),
        )
        for i, bond in enumerate(host_guest.get_bonds())
    ),
    position_matrix=host_guest.get_position_matrix(),
)
print(supramolecule)  # noqa: T201

cg = spd.Spinner(
    step_size=1,
    rotation_step_size=5,
    num_conformers=10,
)

conformer = cg.get_final_conformer(supramolecule)
print(conformer)  # noqa: T201
print(conformer.get_cid(), conformer.get_potential())  # noqa: T201

# Write optimised structure out.
opt_host_guest = host_guest.with_position_matrix(
    conformer.get_position_matrix()
)
opt_host_guest.write("host_multi_guest_out.mol")
