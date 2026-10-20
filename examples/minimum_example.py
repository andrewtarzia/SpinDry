"""Example."""

import pathlib

import stk

import spindry as spd


def main() -> None:
    """Run script."""
    output_dir = pathlib.Path("min_example_output")
    output_dir.mkdir(exist_ok=True)

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
    cage.write(output_dir / "poc.mol")
    stk_guests = (stk.BuildingBlock("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),)

    host_guest = stk.ConstructedMolecule(
        topology_graph=stk.host_guest.Complex(
            host=stk.BuildingBlock.init_from_molecule(cage),
            guests=[stk.host_guest.Guest(i) for i in stk_guests],
        )
    )
    host_guest.write(output_dir / "host_guest.mol")

    supramolecule = spd.SupraMolecule.init_from_stk(host_guest)

    cg = spd.Spinner(
        step_size=0.5,
        rotation_step_size=5,
        num_conformers=10,
    )

    for conformer in cg.get_conformers(supramolecule):
        print(conformer)  # noqa: T201
        print(conformer.get_cid(), conformer.get_potential())  # noqa: T201
        conformer.write_xyz_file(
            output_dir / f"conf_{conformer.get_cid()}.xyz"
        )

    # Write optimised structure out.
    opt_host_guest = host_guest.with_position_matrix(
        conformer.get_position_matrix()
    )
    opt_host_guest.write(output_dir / "host_guest_out.mol")


if __name__ == "__main__":
    main()
