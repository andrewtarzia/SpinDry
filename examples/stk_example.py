import stk
import spindry as spd

# Building a cage from the examples on the stk docs.
bb1 = stk.BuildingBlock(
    smiles='O=CC(C=O)C=O',
    functional_groups=[stk.AldehydeFactory()],
)
bb2 = stk.BuildingBlock(
    smiles='O=CC(Cl)(C=O)C=O',
    functional_groups=[stk.AldehydeFactory()],
)
bb3 = stk.BuildingBlock('NCCN', [stk.PrimaryAminoFactory()])
bb5 = stk.BuildingBlock('NCCCCN', [stk.PrimaryAminoFactory()])

cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        # building_blocks is now a dict, which maps building
        # blocks to the id of the vertices it should be placed
        # on. You can use ranges to specify the ids.
        building_blocks={
            bb1: range(2),
            bb2: (2, 3),
            bb3: (4, 5),
            bb5: range(6, 10),
        },
        optimizer=stk.MCHammer(),
    ),
)
cage.write('stk_example_output/poc.mol')
cage = stk.BuildingBlock.init_from_molecule(cage)
cage_atoms = [
    (atom.get_id(), atom.__class__.__name__)
    for atom in cage.get_atoms()
]

# Build stk guest.
stk_mol = stk.BuildingBlock('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
stk_mol.write('stk_example_output/guest.mol')
stk_mol_atoms = [
    (atom.get_id(), atom.__class__.__name__)
    for atom in stk_mol.get_atoms()
]

host = spd.Molecule.init(
    atoms=(
        spd.Atom(id=i[0], element_string=i[1])
        for i in cage_atoms
    ),
    position_matrix=cage.get_position_matrix(),
)
guest = spd.Molecule.init(
    atoms=(
        spd.Atom(id=i[0], element_string=i[1])
        for i in stk_mol_atoms
    ),
    position_matrix=stk_mol.get_position_matrix(),
)

cg = spd.Spinner(
    step_size=0.5,
    rotation_step_size=5,
    num_conformers=30,
)
for conformer in cg.get_conformers(host, guest):
    print(conformer)
    print(conformer.get_cid(), conformer.get_potential())
    conformer.write_xyz_file(
        f'stk_example_output/conf_{conformer.get_cid()}.xyz'
    )
    cage = cage.with_position_matrix(
        conformer.get_host().get_position_matrix()
    )
    stk_mol = stk_mol.with_position_matrix(
        conformer.get_guest().get_position_matrix()
    )
    complex_mol = stk.ConstructedMolecule(
        topology_graph=stk.host_guest.Complex(cage, stk_mol)
    )
    complex_mol.write(
        f'stk_example_output/conf_{conformer.get_cid()}.mol'
    )