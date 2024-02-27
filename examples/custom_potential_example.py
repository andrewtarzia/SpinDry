import numpy as np  # noqa: D100, INP001
import stk

import spindry as spd

# Building a cage from the examples on the stk docs.
bb1 = stk.BuildingBlock(
    smiles="O=CC(C=O)C=O",
    functional_groups=[stk.AldehydeFactory()],
)
bb2 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])

cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        building_blocks=(bb1, bb2),
    ),
)
cage.write("poc.mol")
# Always ensure initial structures are not entirely on top of each
# other.
stk_guests = (
    (stk.BuildingBlock("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), (0.0, 0.0, 0.0)),
)

host_guest = stk.ConstructedMolecule(
    topology_graph=stk.host_guest.Complex(
        host=stk.BuildingBlock.init_from_molecule(cage),
        guests=[
            stk.host_guest.Guest(i[0], displacement=i[1]) for i in stk_guests
        ],
    )
)

supramolecule = spd.SupraMolecule(
    atoms=(
        spd.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        )
        for atom in host_guest.get_atoms()
    ),
    bonds=(
        spd.Bond(
            id=i,
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            ),
        )
        for i, bond in enumerate(host_guest.get_bonds())
    ),
    position_matrix=host_guest.get_position_matrix(),
)
print(supramolecule)  # noqa: T201


class PotFn(spd.SpdPotential):
    """Scale the size of the guest radii."""

    def __init__(self, guest_scale: float, nonbond_epsilon: float = 5) -> None:
        """Initialize potential class."""
        self._guest_scale = guest_scale
        super().__init__(nonbond_epsilon)

    def compute_potential(self, supramolecule: spd.SupraMolecule) -> float:
        """Compute the potential."""
        component_position_matrices = (
            i.get_position_matrix() for i in supramolecule.get_components()
        )
        component_radii = [
            tuple(j.get_radius() for j in i.get_atoms())
            for i in supramolecule.get_components()
        ]

        component_radii[1] = [
            i * self._guest_scale for i in component_radii[1]
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
    print(round(i, 2), round(conformer.get_potential(), 3))  # noqa: T201


class CentroidFn(spd.Potential):
    """A potential function based on the minimum host-guest distance."""

    def compute_potential(self, supramolecule: spd.SupraMolecule) -> float:
        """Compute the potential."""
        centroids = [i.get_centroid() for i in supramolecule.get_components()]

        return 10 / np.linalg.norm(centroids[1] - centroids[0])


# Do not want to move, just get energy.
cg = spd.Spinner(
    step_size=1,
    rotation_step_size=2,
    num_conformers=100,
    max_attempts=1000,
    potential_function=CentroidFn(),
)
conformer = cg.get_final_conformer(supramolecule)
# Write optimised structure out.
opt_host_guest = host_guest.with_position_matrix(
    conformer.get_position_matrix()
)
opt_host_guest.write("custom_pot_fn_out.mol")
