import stk
import spindry as spd
import numpy as np
from scipy.spatial import distance_matrix


# Building a cage from the examples on the stk docs.
bb1 = stk.BuildingBlock(
    smiles='O=CC(C=O)C=O',
    functional_groups=[stk.AldehydeFactory()],
)
bb2 = stk.BuildingBlock('NCCN', [stk.PrimaryAminoFactory()])

cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        building_blocks=(bb1, bb2),
    ),
)
cage.write('poc.mol')
# Always ensure initial structures are not entirely on top of each
# other.
stk_guests = (
    (stk.BuildingBlock('CN1C=NC2=C1C(=O)N(C(=O)N2C)C'), (0., 0., 0.)),
)

host_guest = stk.ConstructedMolecule(
    topology_graph=stk.host_guest.Complex(
        host=stk.BuildingBlock.init_from_molecule(cage),
        guests=[
            stk.host_guest.Guest(i[0], displacement=i[1])
            for i in stk_guests
        ],
    )
)
host_guest.write('host_multi_guest.mol')

supramolecule = spd.SupraMolecule(
    atoms=(
        spd.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        ) for atom in host_guest.get_atoms()
    ),
    bonds=(
        spd.Bond(
            id=i,
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            )
        ) for i, bond in enumerate(host_guest.get_bonds())
    ),
    position_matrix=host_guest.get_position_matrix(),
)
print(supramolecule)

class PotFn(spd.SpdPotential):
    """
    Scale the size of the guest radii.

    """

    def __init__(self, guest_scale, nonbond_epsilon=5):
        self._guest_scale = guest_scale
        super().__init__(nonbond_epsilon)

    def compute_potential(self, supramolecule):
        component_position_matrices = (
            i.get_position_matrix()
            for i in supramolecule.get_components()
        )
        component_radii = list(
            tuple(j.get_radius() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )

        component_radii[1] = [
            i*self._guest_scale for i in component_radii[1]
        ]
        return self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
        )


for i in np.arange(1, 2.1, 0.1):
    # Do not want to move, just get energy.
    cg = spd.Spinner(
        step_size=0,
        rotation_step_size=0,
        num_conformers=1,
        max_attempts=1,
        potential_function=PotFn(guest_scale=i),
    )

    conformer = cg.get_final_conformer(supramolecule)
    print(round(i, 2), round(conformer.get_potential(), 3))


class YoungPotFn(spd.Potential):
    """
    Repulsion energy function from cgbind.

    `cage_subst_repulsion_func` in `add_substrate.py`.

    https://github.com/duartegroup/cgbind

    """

    def __init__(self, with_attraction):
        self._with_attraction = with_attraction
        super().__init__()

    def compute_potential(self, supramolecule):
        component_position_matrices = list(
            i.get_position_matrix()
            for i in supramolecule.get_components()
        )
        component_radii = list(
            tuple(j.get_radius() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        dist_mat = distance_matrix(
            component_position_matrices[0],
            component_position_matrices[1],
        )
        # Matrix with the pairwise additions of the vdW radii
        sum_vdw_radii = np.add.outer(
            np.array(component_radii[0]),
            np.array(component_radii[1]),
        )

        # Magic numbers derived from fitting potentials to noble
        # gas dimers and plotting against the sum of vdw radii.
        print(
            'warning: these parameters are defined based on radii '
            'defined in cgbind!!!!'
        )
        b_mat = 0.083214 * sum_vdw_radii - 0.003768
        a_mat = 11.576415 * (0.175541 * sum_vdw_radii + 0.316642)
        exponent_mat = -(dist_mat / b_mat) + a_mat

        energy_mat = np.exp(exponent_mat)
        energy = np.sum(energy_mat)

        # E is negative for favourable binding but this is a purely
        # repulsive function so subtract a number.. which is determined
        # from the best classifier for 102 binding affinities (see
        # cgbind paper) 0.4 kcal mol-1.
        if self._with_attraction:
            return energy - 0.4 * len(component_radii[1])
        return energy


for wt in [True, False]:
    # Do not want to move, just get energy.
    cg = spd.Spinner(
        step_size=0,
        rotation_step_size=0,
        num_conformers=1,
        max_attempts=1,
        potential_function=YoungPotFn(wt),
    )

    conformer = cg.get_final_conformer(supramolecule)
    print(
        f'T Young potential (attr={wt}): ',
        round(conformer.get_potential(), 3),
    )
