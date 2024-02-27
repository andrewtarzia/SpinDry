import os

import numpy as np
import pytest

import spindry as spd


def test_molecule_get_position_matrix(
    molecule: spd.Molecule,
    position_matrix: np.ndarray,
) -> None:
    assert np.all(
        np.allclose(
            position_matrix,
            molecule.get_position_matrix(),
        )
    )


def test_molecule_with_position_matrix(
    molecule: spd.Molecule,
    position_matrix2: np.ndarray,
) -> None:
    test = molecule.with_position_matrix(position_matrix2)
    assert np.all(
        np.allclose(
            position_matrix2,
            test.get_position_matrix(),
        )
    )


def test_molecule_with_displacement(
    molecule: spd.Molecule,
    displacement: np.ndarray,
    displaced_position_matrix: np.ndarray,
) -> None:
    test = molecule.with_displacement(displacement)
    assert np.all(
        np.allclose(
            displaced_position_matrix,
            test.get_position_matrix(),
        )
    )


@pytest.fixture()
def path(tmpdir) -> str:  # noqa: ANN001
    return os.path.join(tmpdir, "molecule.xyz")  # noqa: PTH118


def test_molecule_write_xyz_file(molecule: spd.Molecule, path: str) -> None:
    molecule.write_xyz_file(path)
    content = molecule._write_xyz_content()  # noqa: SLF001
    with open(path) as f:
        test_lines = f.readlines()

    assert "".join(test_lines) == "".join(content)


def test_molecule_get_atoms(molecule: spd.Molecule, atoms: int) -> None:
    for test, atom in zip(molecule.get_atoms(), atoms):
        assert test.get_id() == atom.get_id()
        assert test.get_element_string() == atom.get_element_string()


def test_molecule_get_centroid(
    molecule: spd.Molecule,
    position_matrix: np.ndarray,
    centroid: np.ndarray,
) -> None:
    test = molecule.with_position_matrix(position_matrix)
    assert np.all(
        np.allclose(
            centroid,
            test.get_centroid(),
            atol=1e-6,
        )
    )


def test_molecule_get_num_atoms(
    molecule: spd.Molecule, num_atoms: int
) -> None:
    assert molecule.get_num_atoms() == num_atoms
