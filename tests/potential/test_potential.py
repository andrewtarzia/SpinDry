import numpy as np


def test_nonbond_potential(
    spdpotential, distances, nonbond_potentials, nb_mins,
):

    for i, d in enumerate(distances):
        test = spdpotential._nonbond_potential(
            distance=d,
            sigmas=np.array(1.2),
        )
        assert np.isclose(test, nonbond_potentials[i], atol=1E-5)

    for i, _sigma in enumerate([1.2, 1.4, 1.8, 1.6]):
        test = spdpotential._nonbond_potential(
            distance=distances,
            sigmas=np.array(_sigma),
        )
        assert np.argmin(test**2) == nb_mins[i]


def test_combinations(spdpotential, radii_combinations):
    for combo in radii_combinations:
        test = spdpotential._mixing_function(combo[0], combo[1])

        assert test == combo[2]
