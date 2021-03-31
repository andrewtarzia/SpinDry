import spindry as spd


host = spd.Molecule('eg_host.xyz')
guest = spd.Molecule('eg_guest.xyz')

print(host, guest)

conformer_generator = spd.Spinner(host=host, guest=guest)

for cid, conformer in enumerate(conformer_generator):
    conformer.write(f'min_example_output/conf_{cid}.xyz')
