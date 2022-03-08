import numpy as np
from ..utilities import is_equivalent_spd_molecule


def test_opt(spinner, smolecule, final_pos_mat, final_potential):
    test = spinner.get_final_conformer(smolecule)
    print(test.get_position_matrix())
    assert np.all(np.allclose(
        final_pos_mat,
        test.get_position_matrix(),
    ))
    assert np.isclose(
        spinner._compute_potential(test),
        final_potential
    )


def test_opt_spd(
    spd_spinner,
    spd_host,
    spd_guest,
    spd_supramolecule,
):
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
    spd_spinner,
    spd_host,
    spd_guest,
    spd_supramolecule,
):
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


def test_opt_test_move(spinner):
    # Do not test random component.
    assert spinner._test_move(curr_pot=-1, new_pot=-2)


def test_opt_setcomp1(
    spinner,
    smolecule_components,
    final_comp_pos_mat1,
    movable_components1,
):
    test = spinner.get_final_conformer(
        smolecule_components, movable_components1,
    )
    print(test.get_position_matrix())
    assert np.all(np.allclose(
        final_comp_pos_mat1,
        test.get_position_matrix(),
    ))


def test_opt_setcomp2(
    spinner,
    smolecule_components,
    final_comp_pos_mat2,
    movable_components2,
):
    test = spinner.get_final_conformer(
        smolecule_components, movable_components2,
    )
    print(test.get_position_matrix())
    assert np.all(np.allclose(
        final_comp_pos_mat2,
        test.get_position_matrix(),
    ))
