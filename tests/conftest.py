import pytest
import numpy as np
import spindry as spd


@pytest.fixture(
    params=(
        (spd.Atom(id=0, element_string='N'), 0, 'N'),
        (spd.Atom(id=65, element_string='P'), 65, 'P'),
        (spd.Atom(id=2, element_string='C'), 2, 'C'),
    )
)
def atom_info(request):
    return request.param


@pytest.fixture
def atoms():
    return [
        spd.Atom(0, 'C'), spd.Atom(1, 'C'), spd.Atom(2, 'C'),
        spd.Atom(3, 'C'), spd.Atom(4, 'C'), spd.Atom(5, 'C'),
    ]


@pytest.fixture
def position_matrix():
    return np.array([
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 10, 0],
        [1, 10, 0],
        [-1, 10, 0],
    ])


@pytest.fixture
def position_matrix2():
    return np.array([
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 20, 0],
        [1, 20, 0],
        [-1, 20, 0],
    ])


@pytest.fixture
def centroid():
    return np.array([0, 5.5, 0])


@pytest.fixture
def molecule(atoms, position_matrix):
    return spd.Molecule.init(
        atoms=atoms,
        position_matrix=position_matrix
    )


@pytest.fixture
def nonbond_potentials():
    return [
        34.559999999999995, 4.319999999999999, 1.2799999999999998,
        0.5399999999999999, 0.27647999999999995, 0.15999999999999998,
        0.10075801749271138,
    ]


@pytest.fixture
def nonbonded_potential():
    return 122.51087809973211


@pytest.fixture
def position_matrix3():
    return np.array([
        [0, -4, 0],
        [1, -4, 0],
        [-1, -4, 0],
        [0, 5, 0],
        [1, 5, 0],
        [-1, 5, 0],
    ])


@pytest.fixture
def spinner():
    return spd.Spinner(
        step_size=0.5,
        rotation_step_size=5,
        num_conformers=10,
    )


@pytest.fixture
def host_position_matrix():
    return np.array([
        [1, 1, 0],
        [-1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
    ])


@pytest.fixture
def guest_position_matrix():
    return np.array([
        [0, 0.5, 0],
        [0, -0.5, 0],
    ])


@pytest.fixture
def guest_position_matrix2():
    return np.array([
        [0.5, 0, 0],
        [-0.5, 0, 0],
    ])