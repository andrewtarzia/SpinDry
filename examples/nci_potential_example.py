"""Example."""

import os
import pathlib
import shutil
import subprocess as sp
import time

import matplotlib.pyplot as plt
import numpy as np
import rdkit.Chem.AllChem as rdkit  # noqa: N813
import stk

import spindry as spd


class NciPlotFunction:
    """A potential function based on the minimum host-guest distance.

    Here we show that as long as you provide a `compute_potential` method,
    then any class works!

    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        nciplot_path: pathlib.Path,
    ) -> None:
        """Initialize NCIPlot function."""
        self._output_dir = output_dir
        self._nciplot_path = nciplot_path

    def _calculate_energy_single(
        self,
        output_file: pathlib.Path,
        ispromol: bool,
        supra: bool,
    ) -> tuple[float, float, float]:
        """Calculate the NCI energy given the NCIPLOT output."""
        with output_file.open("r") as f:
            output = f.readlines()

        sis = output.index(
            "               RANGE INTEGRATION DATA                            "
            "     \n"
        )

        polar_indices = (8, 15)
        vdw_indices = (33, 40)
        rep_indices = (58, 65)

        displacement = 18 if "Warning!" in output[sis + 3] else 0

        polar = np.array(
            [
                float(i.split(":")[1].split()[0])
                for i in output[
                    sis + polar_indices[0] + displacement : sis
                    + polar_indices[1]
                    + displacement
                ]
            ]
        )

        vdw = np.array(
            [
                float(i.split(":")[1].split()[0])
                for i in output[
                    sis + vdw_indices[0] + displacement : sis
                    + vdw_indices[1]
                    + displacement
                ]
            ]
        )
        rep = np.array(
            [
                float(i.split(":")[1].split()[0])
                for i in output[
                    sis + rep_indices[0] + displacement : sis
                    + rep_indices[1]
                    + displacement
                ]
            ]
        )

        NCI_index_dict = {  # noqa: N806
            "Strong": polar[[0, 5, 1, 6, 2, 3, 4]],
            "Weak": vdw[[0, 5, 1, 6, 2, 3, 4]],
            "Repulsion": rep[[0, 5, 1, 6, 2, 3, 4]],
        }

        if ispromol:
            if supra:
                E_polar = -1381.3065 * np.array(NCI_index_dict["Strong"][3])  # noqa: N806
                E_vdw = 424.2966 * np.array(  # noqa: N806
                    NCI_index_dict["Weak"][1]
                ) - 109.60437 * np.array(
                    NCI_index_dict["Weak"][0]
                    + NCI_index_dict["Weak"][6] ** (1 / 3)
                )
            else:
                E_polar = -np.array(  # noqa: N806
                    27.424896 * np.power(NCI_index_dict["Strong"][1], 0.333)
                    + 2759.675 * (NCI_index_dict["Strong"][4])
                )
                E_vdw = -np.array(  # noqa: N806
                    -79.92235 * np.power(NCI_index_dict["Weak"][3], 0.333)
                    + 50.483402 * np.power(NCI_index_dict["Weak"][0], 0.5)
                )
        elif supra:
            E_polar = -19199.29366455 * NCI_index_dict["Strong"][4]  # noqa: N806
            E_vdw = (  # noqa: N806
                -396.6232038 * NCI_index_dict["Weak"][5] ** (1 / 3)
                - 51.94821914 * NCI_index_dict["Weak"][0]
            )
        else:
            E_polar = -np.array(3399.1965 * NCI_index_dict["Strong"][2])  # noqa: N806
            E_vdw = -np.array(  # noqa: N806
                -811.5827 * NCI_index_dict["Weak"][0] ** 2
                + 115.258 * np.power(NCI_index_dict["Weak"][5], 0.333)
                + 3399.1965 * NCI_index_dict["Weak"][2]
            )

        return E_polar + E_vdw, E_polar, E_vdw

    def compute_potential(self, supramolecule: spd.SupraMolecule) -> float:
        """Compute the potential."""
        output_dir = self._output_dir.resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True)
        init_dir = pathlib.Path.cwd()
        os.chdir(output_dir)
        try:
            supramolecule.write_xyz_file("input.xyz")
            print(supramolecule)
            fragment_list = []
            for comp in supramolecule.get_components():
                print(comp)
                fragment_list.append([i.get_id() for i in comp.get_atoms()])
            print(fragment_list)

            # Run 1.
            in_file = pathlib.Path("run1.in")
            dat_file = pathlib.Path("run1.dat")
            raise SystemExit("include charges")
            """
            repulsion part - still a problem - maybe start with spindry

            if you use a constant set of parameters (with INTERMOLECULAR, there was a default), you get good consistency of the meaning of energies

            charge correction is from a regressor for charge prediction from xtb, comes from
            """

            input_str = (
                "1\n"
                "input.xyz\n"
                "ONAME run1\n"
                "INTEGRATE\n"
                "INTERMOLECULAR\n"
                "FINE\n"
                "SUPRA\n"
                "CHARGE {num} {num}\n"
                # "RANGE 3 # Integration is performed in three ranges.\n"
                # "-0.2 -0.02\n"
                # "-0.02 0.02\n"
                # "0.02 0.2\n"
            )

            with in_file.open("w") as f:
                f.write(input_str)

            output_file = pathlib.Path("run1.out")
            err_file = pathlib.Path("run1.err")
            cmd = [self._nciplot_path, in_file]
            with output_file.open("w") as f_out, err_file.open("w") as f_err:
                sp.run(  # noqa: PLW1510, S603
                    cmd,
                    input=None,
                    stdout=f_out,
                    stderr=f_err,
                    text=True,
                )

            if not dat_file.exists():
                raise RuntimeError

            e_sum, e_polar, e_vdw = self._calculate_energy_single(
                output_file=output_file,
                ispromol=True,
                supra=True,
            )
            print(e_sum, e_polar, e_vdw)

            # fig, ax = plt.subplots(figsize=(8, 5))

            # signed_density = []
            # rdgrad = []
            # with pathlib.Path("run1.dat").open("r") as f:
            #     for line in f:
            #         x, y = line.strip().split()
            #         signed_density.append(x)
            #         rdgrad.append(y)

            # ax.scatter(signed_density, rdgrad, marker="o", rasterized=True)

            # ax.tick_params(axis="both", which="major", labelsize=16)
            # ax.set_xlabel(
            #     r"sgn$\mathbf{(\lambda_2)}$ $\times$ $\mathbf{\rho(r)}$ (a.u)",
            #     fontsize=16,
            # )
            # ax.set_ylabel("$s$ [a.u.]", fontsize=16)
            # ax.set_xlim(-0.2, 0.2)

            # fig.tight_layout()
            # fig.savefig("run1.png", dpi=360, bbox_inches="tight")
            # plt.close()

        finally:
            os.chdir(init_dir)

        return e_sum
        raise SystemExit
        centroids = [i.get_centroid() for i in supramolecule.get_components()]

        # set ylabel ’s(a.u.)’ font "Helvetica, 30"
        # set xlabel ’sign({/Symbol l}_2){/Symbol r}(a.u.)’ font "Helvetica, 30"
        # set pm3d map
        # # Define a colour gradient palette used by pm3d
        # set palette defined (-0.07 "blue",0.00 "green", 0.07 "red")
        # set format y "% .2f"
        # set format x "% .2f"
        # set format cb "% -.2f"

        # -0.06,0.01,0.06 nomirror font "Helvetica"
        # set xrange [-0.07:0.07]
        # set yrange [0.0:1.0]
        # # set the range of values which are colored using the current palette
        # set cbrange [-0.06:0.06]
        # plot ’AT.dat’ u 1:2:1 w p lw 6 palette t ’’

        return 10 / np.linalg.norm(centroids[1] - centroids[0])


def main() -> None:
    """Run script."""
    output_dir = pathlib.Path("nci_potential_example")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "nci_test").mkdir(exist_ok=True)

    nciplot_path = pathlib.Path(
        "/home/atarzia/thesource/NCIPLOT-4.4/src_4.4/nciplot"
    )
    os.environ["NCIPLOT_HOME"] = "/home/atarzia/thesource/NCIPLOT-4.4/"

    # Building a cage from the examples on the stk docs.
    bb1 = stk.BuildingBlock(
        smiles="O=CC(C=O)C=O",
        functional_groups=[stk.AldehydeFactory()],
    )
    bb2 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])

    cage = stk.ConstructedMolecule(
        topology_graph=stk.cage.FourPlusSix(
            building_blocks=(bb1, bb2),
            optimizer=stk.MCHammer(),
        ),
    )
    # Optimize with the MMFF force field.

    rdkit_bb = cage.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_bb)
    rdkit.MMFFOptimizeMolecule(rdkit_bb)

    # stk molecules are immutable. with_position_matrix returns a
    # a clone, holding the new position matrix.
    cage = cage.with_position_matrix(
        position_matrix=rdkit_bb.GetConformer().GetPositions(),
    )
    cage.write(output_dir / "poc.mol")
    # Always ensure initial structures are not entirely on top of each
    # other.
    stk_guests = ((stk.BuildingBlock("c1ccccc1"), (0.0, 0.0, 4.0)),)

    host_guest = stk.ConstructedMolecule(
        topology_graph=stk.host_guest.Complex(
            host=stk.BuildingBlock("C12C=CC=C3C=1C1=C(C=CC=C1C=C3)C=C2"),
            guests=[
                stk.host_guest.Guest(i[0], displacement=i[1])
                for i in stk_guests
            ],
        )
    )
    host_guest.write(output_dir / "host_guest.mol")

    supramolecule = spd.SupraMolecule.init_from_stk(host_guest)

    # Do not want to move, just get energy.
    cg = spd.Spinner(
        step_size=0.5,
        rotation_step_size=0.2,
        num_conformers=100,
        max_attempts=100,
        potential_function=NciPlotFunction(
            output_dir=output_dir / "nci_test",
            nciplot_path=nciplot_path,
        ),
    )
    energies = []
    count = 0
    prev_time = time.time()
    for conformer in cg.get_conformers(supramolecule):
        print(time.time() - prev_time)
        print(count)
        count += 1
        print(conformer)  # noqa: T201
        print(conformer.get_cid(), conformer.get_potential())  # noqa: T201
        energies.append(conformer.get_potential())
        conformer.write_xyz_file(
            output_dir / f"conf_{conformer.get_cid()}.xyz"
        )
        prev_time = time.time()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(energies)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("conformer", fontsize=16)
    ax.set_ylabel("e_sum [kJ/mol]", fontsize=16)

    fig.tight_layout()
    fig.savefig("run1.png", dpi=360, bbox_inches="tight")
    plt.close()

    # Write optimised structure out.
    opt_host_guest = host_guest.with_position_matrix(
        conformer.get_position_matrix()
    )
    opt_host_guest.write(output_dir / "nciplot_opt.mol")
    raise SystemExit("include the other terms, and the stacking distance")

    raise SystemExit("plot energies")
    raise SystemExit


if __name__ == "__main__":
    main()
