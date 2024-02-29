import pathlib

import numpy as np
import spindry as spd
import stk


def test_stk_spinner() -> None:
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

    complex_opt = stk.ConstructedMolecule(
        topology_graph=stk.host_guest.Complex(
            host=stk.BuildingBlock.init_from_molecule(cage),
            guests=(guest1, guest2),
            optimizer=stk.Spinner(),
        ),
    )

    complex_unopt = stk.ConstructedMolecule(
        topology_graph=stk.host_guest.Complex(
            host=stk.BuildingBlock.init_from_molecule(cage),
            guests=(guest1, guest2),
        ),
    )
    supramolecule = spd.SupraMolecule(
        atoms=(
            spd.Atom(
                id=atom.get_id(),
                element_string=atom.__class__.__name__,
            )
            for atom in complex_unopt.get_atoms()
        ),
        bonds=(
            spd.Bond(
                id=i,
                atom_ids=(
                    bond.get_atom1().get_id(),
                    bond.get_atom2().get_id(),
                ),
            )
            for i, bond in enumerate(complex_unopt.get_bonds())
        ),
        position_matrix=complex_unopt.get_position_matrix(),
    )
    # Run optimization.
    optimizer = spd.Spinner(
        potential_function=spd.SpdPotential(
            nonbond_epsilon=5.0,
        ),
        step_size=1.5,
        rotation_step_size=5.0,
        num_conformers=50,
        max_attempts=1000,
        beta=2.0,
        random_seed=1000,
    )
    conformer = optimizer.get_final_conformer(supramolecule)
    new_complex = complex_unopt.with_position_matrix(
        position_matrix=conformer.get_position_matrix(),
    )

    assert np.all(
        np.equal(
            complex_opt.get_position_matrix(),
            new_complex.get_position_matrix(),
        )
    )

    known = stk.BuildingBlock.init_from_file(
        pathlib.Path(__file__).resolve().parent / "spinner.mol"
    )

    assert np.all(
        np.isclose(
            known.get_position_matrix(),
            new_complex.get_position_matrix(),
            atol=1e-2,
        )
    )
