import numpy as np


def test_nonbond_potential(
    spinner, distances, nonbond_potentials, nb_mins,
):

    for i, d in enumerate(distances):
        test = spinner._nonbond_potential(
            distance=d,
            sigmas=np.array(1.2),
        )
        assert np.isclose(test, nonbond_potentials[i], atol=1E-5)

    for i, _sigma in enumerate([1.2, 1.4, 1.8, 1.6]):
        test = spinner._nonbond_potential(
            distance=distances,
            sigmas=np.array(_sigma),
        )
        assert np.argmin(test**2) == nb_mins[i]

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


def test_combinations(spinner, radii_combinations):
    for combo in radii_combinations:
        test = spinner._mixing_function(combo[0], combo[1])

        assert test == combo[2]
