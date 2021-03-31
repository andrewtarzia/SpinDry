import numpy as np


def test_opt_nonbond_potential(spinner, nonbond_potentials):
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = spinner._nonbond_potential(distance=d)
        assert np.isclose(test, nonbond_potentials[i], atol=1E-5)


def test_opt_compute_nonbonded_potential(
    spinner,
    host_position_matrix,
    guest_position_matrix,
    guest_position_matrix2,
    nonbonded_potential,
):
    test = spinner._compute_nonbonded_potential(
        host_position_matrix,
        guest_position_matrix,
    )
    assert test == nonbonded_potential
    test = spinner._compute_nonbonded_potential(
        host_position_matrix,
        guest_position_matrix2,
    )
    assert test == nonbonded_potential

def test_opt_translate_atoms_along_vector(
    spinner, molecule, position_matrix, position_matrix3
):
    molecule = molecule.with_position_matrix(position_matrix)
    new_molecule = spinner._translate_atoms_along_vector(
        mol=molecule,
        vector=np.array([0, 5, 0])
    )
    print(position_matrix, new_molecule.get_position_matrix())
    assert np.all(np.equal(
        position_matrix3,
        new_molecule.get_position_matrix(),
    ))
    new_molecule = spinner._translate_atoms_along_vector(
        mol=new_molecule,
        vector=np.array([0, -5, 0])
    )
    print(position_matrix, new_molecule.get_position_matrix())
    assert np.all(np.equal(
        position_matrix,
        new_molecule.get_position_matrix(),
    ))


def test_opt_test_move(spinner):
    # Do not test random component.
    assert spinner._test_move(curr_pot=-1, new_pot=-2)
