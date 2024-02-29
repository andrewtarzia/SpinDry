from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import spindry as spd


def test_nonbond_potential(
    spdpotential: spd.SpdPotential,
    distances: np.ndarray,
    nonbond_potentials: list[float],
    nb_mins: list[int],
) -> None:
    for i, d in enumerate(distances):
        test = spdpotential._nonbond_potential(  # noqa: SLF001
            distance=d,
            sigmas=np.array(1.2),
        )
        assert np.isclose(test, nonbond_potentials[i], atol=1e-5)

    for i, _sigma in enumerate([1.2, 1.4, 1.8, 1.6]):
        test = spdpotential._nonbond_potential(  # noqa: SLF001
            distance=distances,
            sigmas=np.array(_sigma),
        )
        assert np.argmin(test**2) == nb_mins[i]
