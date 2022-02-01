import pytest
import numpy as np
import spindry as spd


@pytest.fixture(
    params=(
        (spd.Atom(id=0, element_string='N'), 0, 'N', 1.4882656711484),
        (spd.Atom(id=65, element_string='P'), 65, 'P', 1.925513788),
        (spd.Atom(id=2, element_string='C'), 2, 'C', 1.60775485914852),
    )
)
def atom_info(request):
    return request.param


@pytest.fixture(
    params=(
        (spd.Bond(id=0, atom_ids=(0, 1)), 0, 0, 1),
        (spd.Bond(id=65, atom_ids=(2, 3)), 65, 2, 3),
        (spd.Bond(id=2, atom_ids=(3, 4)), 2, 3, 4),
        (spd.Bond(id=3, atom_ids=(0, 9)), 3, 0, 9),
    )
)
def bond_info(request):
    return request.param


@pytest.fixture
def atoms():
    return [spd.Atom(0, 'C'), spd.Atom(1, 'C')]


@pytest.fixture
def position_matrix():
    return np.array([[0, 0, 0], [0, 1.5, 0]])


@pytest.fixture
def position_matrix2():
    return np.array([[0, 0, 0], [0, 3, 0]])


@pytest.fixture
def displacement():
    return np.array([0, 1, 0])


@pytest.fixture
def displaced_position_matrix():
    return np.array([[0, 1, 0], [0, 2.5, 0]])


@pytest.fixture
def centroid():
    return np.array([0, 0.75, 0])


@pytest.fixture
def num_atoms():
    return 2


@pytest.fixture
def molecule(atoms, position_matrix):
    return spd.Molecule(
        atoms=atoms,
        bonds=[],
        position_matrix=position_matrix
    )


@pytest.fixture
def components(atoms, position_matrix):
    return (
        spd.Molecule(
            atoms=(atoms[0], ),
            bonds=[],
            position_matrix=position_matrix[0]
        ),
        spd.Molecule(
            atoms=(atoms[1], ),
            bonds=[],
            position_matrix=position_matrix[1]
        ),
    )


@pytest.fixture
def smolecule(atoms, position_matrix):
    return spd.SupraMolecule(
        atoms=atoms,
        bonds=[],
        position_matrix=position_matrix,
    )


@pytest.fixture
def distances():
    return np.array([1, 1.2, 1.4, 1.6, 1.8, 2.0, 4])


@pytest.fixture
def nonbond_potentials():
    return [
        29.650582241279984, 0.0, -1.1965106134640058,
        -0.7315108180046076, -0.40042074284821816,
        -0.22239608831999993, -0.0036423427949999992
    ]


@pytest.fixture
def nb_mins():
    return [1, 2, 4, 3]


@pytest.fixture
def final_potential():
    return -0.0564393681661353


@pytest.fixture
def final_pos_mat():
    return np.array([
        [-0.02514959, -0.50457139, -0.20141202],
        [ 1.31683193,  2.57903968,  0.21028248]
    ])


@pytest.fixture
def spinner():
    return spd.Spinner(
        step_size=0.5,
        rotation_step_size=5,
        num_conformers=50,
    )


@pytest.fixture
def spdpotential():
    return spd.SpdPotential()


@pytest.fixture
def radii_combinations():
    return [
        (1.24235230881914, 1.24235230881914, 1.24235230881914),
        (1.24235230881914, 2.056812828, 1.64958256840957),
        (1.75038091053539, 1.24235230881914, 1.496366609677265),
        (1.60775485914852, 1.4882656711484, 1.54801026514846),
    ]
