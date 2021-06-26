import spindry as spd

host = spd.Molecule('eg_host.xyz')
guest = spd.Molecule('eg_guest.xyz')
guest2 = spd.Molecule('eg_guest2.xyz')
host = host.with_centroid([0, 0, 0])
guest = guest.with_centroid([0, 0, 0])
guest2 = guest2.with_centroid([0, 0, 0])
print(host, guest, guest2)

cg = spd.Spinner(
    step_size=0.5,
    rotation_step_size=5,
    num_conformers=10,
)

for conformer in cg.get_conformers(host, [guest, guest2]):
    print(conformer)
    print(conformer.get_cid(), conformer.get_potential())
    conformer.write_xyz_file(
        f'multi_example_output/conf_{conformer.get_cid()}.xyz'
    )
