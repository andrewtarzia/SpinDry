import stk

bb1 = stk.BuildingBlock(
    smiles="NCCN",
    functional_groups=[stk.PrimaryAminoFactory()],
)
bb2 = stk.BuildingBlock(
    smiles="O=CC(C=O)C=O",
    functional_groups=[stk.AldehydeFactory()],
)
guest1 = stk.host_guest.Guest(
    building_block=stk.BuildingBlock("c1ccccc1"),
)
guest2 = stk.host_guest.Guest(
    building_block=stk.BuildingBlock("C1CCCCC1"),
)
cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        building_blocks=(bb1, bb2),
        optimizer=stk.MCHammer(),
    ),
)

hgcomplex = stk.ConstructedMolecule(
    topology_graph=stk.host_guest.Complex(
        host=stk.BuildingBlock.init_from_molecule(cage),
        guests=(guest1, guest2),
        optimizer=stk.Spinner(),
    ),
)
hgcomplex.write("spinner.mol")
