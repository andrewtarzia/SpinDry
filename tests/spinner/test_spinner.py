import mchammer as mch
import numpy as np
import spindry as spd

from tests.utilities import is_equivalent_spd_molecule


def test_opt(
    spinner: spd.Spinner,
    smolecule: mch.Molecule,
    final_pos_mat: np.ndarray,
    final_potential: float,
) -> None:
    test = spinner.get_final_conformer(smolecule)
    print(test.get_position_matrix(), final_pos_mat)
    assert np.all(
        np.allclose(
            final_pos_mat,
            test.get_position_matrix(),
        )
    )
    print(spinner.compute_potential(test), final_potential)
    assert np.isclose(spinner.compute_potential(test), final_potential)


def test_opt_spd(
    spd_spinner: spd.Spinner,
    spd_host: mch.Molecule,
    spd_guest: mch.Molecule,
    spd_supramolecule: spd.SupraMolecule,
) -> None:
    test = spd_spinner.get_final_conformer(spd_supramolecule)
    is_equivalent_spd_molecule(test, spd_supramolecule)
    for i, comp in enumerate(list(test.get_components())):
        if i == 0:
            test_host = spd_host.with_position_matrix(
                comp.get_position_matrix()
            )
            is_equivalent_spd_molecule(test_host, spd_host)
        elif i == 1:
            test_guest = spd_guest.with_position_matrix(
                comp.get_position_matrix()
            )
            is_equivalent_spd_molecule(test_guest, spd_guest)


def test_opt_spd_components(
    spd_spinner: spd.Spinner,
    spd_host: mch.Molecule,
    spd_guest: mch.Molecule,
    spd_supramolecule: spd.SupraMolecule,
) -> None:
    test = spd_spinner.get_final_conformer(spd_supramolecule)
    is_equivalent_spd_molecule(test, spd_supramolecule)
    for i, comp in enumerate(list(test.get_components())):
        if i == 0:
            test_host = spd_host.with_position_matrix(
                comp.get_position_matrix()
            )
            is_equivalent_spd_molecule(test_host, spd_host)
        elif i == 1:
            test_guest = spd_guest.with_position_matrix(
                comp.get_position_matrix()
            )
            is_equivalent_spd_molecule(test_guest, spd_guest)


def test_opt_setcomp1(
    spinner: spd.Spinner,
    smolecule_components: spd.SupraMolecule,
    final_comp_pos_mat1: np.ndarray,
    movable_components1: tuple,
) -> None:
    test = spinner.get_final_conformer(
        smolecule_components,
        movable_components1,
    )
    print(test.get_position_matrix(), final_comp_pos_mat1)
    assert np.all(
        np.allclose(
            final_comp_pos_mat1,
            test.get_position_matrix(),
        )
    )


def test_opt_setcomp2(
    spinner: spd.Spinner,
    smolecule_components: spd.SupraMolecule,
    final_comp_pos_mat2: np.ndarray,
    movable_components2: tuple,
) -> None:
    test = spinner.get_final_conformer(
        smolecule_components,
        movable_components2,
    )
    print(test.get_position_matrix(), final_comp_pos_mat2)
    assert np.all(
        np.allclose(
            final_comp_pos_mat2,
            test.get_position_matrix(),
        )
    )
