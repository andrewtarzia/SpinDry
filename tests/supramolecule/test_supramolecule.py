import numpy as np


def test_smolecule_define_components(smolecule, components):
    tests = list(smolecule.get_components())

    assert len(tests) == len(components)
    for test, comp in zip(tests, components):
        for a1, a2 in zip(test.get_atoms(), comp.get_atoms()):
            assert a1.get_id() == a2.get_id()
            assert a1.get_element_string() == a2.get_element_string()

        assert np.all(np.allclose(
            test.get_position_matrix(),
            comp.get_position_matrix(),
        ))
