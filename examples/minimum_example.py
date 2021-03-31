import spindry as spd


host = spd.Molecule('eg_host.xyz')
guest = spd.Molecule('eg_guest.xyz')

print(host, guest)

cg = spd.Spinner(
    step_size=0.5,
    rotation_step_size=5,
    num_conformers=10,
)

for conformer in cg.get_conformers(host, guest):
    print(conformer)
    print(conformer.get_cid(), conformer.get_potential())
    conformer.write_xyz_file(
        f'min_example_output/conf_{conformer.get_cid()}.xyz'
    )
