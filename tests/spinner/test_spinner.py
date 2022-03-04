import numpy as np


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


def test_opt_test_move(spinner):
    # Do not test random component.
    assert spinner._test_move(curr_pot=-1, new_pot=-2)


def test_opt_setcomp(
    spinner,
    smolecule,
    final_pos_mat2,
    final_potential2,
    movable_components,
):
    test = spinner.get_final_conformer(smolecule, movable_components)
    print(test.get_position_matrix())
    assert np.all(np.allclose(
        final_pos_mat2,
        test.get_position_matrix(),
    ))
    assert np.isclose(
        spinner._compute_potential(test),
        final_potential2
    )
