.. toctree::
  :hidden:
  :maxdepth: 0
  :caption: SpinDry

  SpinDry <spd>

.. toctree::
  :hidden:
  :maxdepth: 0
  :caption: Modules:

  Modules <modules>


Introduction
------------

| GitHub: https://www.github.com/andrewtarzia/SpinDry

:mod:`.spindry` is a Monte Carlo-based host-guest conformer generator using
cheap and unphysical potentials.

Please submit an issue with any questions or bugs!

``SpinDry`` uses the Monte-Carlo/molecule interface provided by my other code
``MCHammer`` (https://github.com/andrewtarzia/MCHammer).



Usage with *stk*
----------------

:mod:`.spindry` optimisers (``Spinner``) are available by default in ``stk``
as:

.. code-block:: python

  import stk

  host = stk.ConstructedMolecule(
      topology_graph=stk.cage.FourPlusSix(
          building_blocks=(
              stk.BuildingBlock(
                  smiles='NC1CCCCC1N',
                  functional_groups=[
                      stk.PrimaryAminoFactory(),
                  ],
              ),
              stk.BuildingBlock(
                  smiles='O=Cc1cc(C=O)cc(C=O)c1',
                  functional_groups=[stk.AldehydeFactory()],
              ),
          ),
          optimizer=stk.MCHammer(),
      ),
  )
  guest1 = stk.host_guest.Guest(
      building_block=stk.BuildingBlock('BrBr'),
      displacement=(0., 3., 0.),
  )
  guest2 = stk.host_guest.Guest(
      building_block=stk.BuildingBlock('C1CCCC1'),
  )

  hgcomplex = stk.ConstructedMolecule(
      topology_graph=stk.host_guest.Complex(
          host=stk.BuildingBlock.init_from_molecule(host),
          guests=(guest1, guest2),
          optimizer=stk.Spinner(),
      ),
  )



Installation
------------

Install using pip:

.. code-block:: bash

  pip install spindry


Algorithm
---------

SpinDry implements a simple Metropolis Monte-Carlo algorithm to translate and
rotate the guest molecules.
All atom positions/bond lengths within the host and guest are kept rigid and
do not contribute to the potential energy.
The algorithm uses, by default, a simple Lennard-Jones nonbonded potential to
define the potential energy surface such that steric clashes are avoided. Atom
radii are taken from STREUSSEL (https://github.com/hmsoregon/STREUSEL).
Custom potential functions can also be defined now -- see
``examples/custom_potential_function.py``.

The default MC algorithm is as follows:

For ``step`` in *N* steps:

    1. Define a translation of the guest by a random unit-vector and a random
    [-1, 1) step along the that vector.

    2. Define a rotation of the guest by a random [-1, 1) * ``rotation_step_size``
    angle and a random unit axis.

    3. Compute system potential ``U_nb``:

        ``U_nb`` is the nonbonded potential, defined by the Lennard-Jones
        potential:

            ``U_nb = sum_i,j (epsilon_nb * ((sigma / r_ij)^12 - (sigma / r_ij)^6))``,
            where ``epsilon_nb`` defines the strength of the potential,
            ``sigma`` defines the position where the potential becomes
            repulsive and ``r_ij`` is the pairwise distance between atoms
            ``i`` and ``j``.

    4. Accept or reject move:

        Accept if ``U_i`` < ``U_(i-1)`` or ``exp(-beta(U_i - U_(i-1))`` >
        ``R``, where ``R`` is a random number [0, 1) and ``beta`` is the
        inverse Boltzmann temperature.
        Reject otherwise.

    5. If ``num_conformers`` is met, quit.

Examples
--------

The workflow for a porous organic cage built using *stk*
(<https://stk.readthedocs.io/>) is shown in ``examples/`` for a single guest
and multiple guests.

The Spinner class yields a ``SupraMolecule`` conformer. Only conformers that
pass the MC conditions are yielded. The examples in ``examples`` show how to
access the structures of these conformers as ``.xyz`` files or `stk` molecules.

Contributors and Acknowledgements
---------------------------------

I developed this code as a post doc in the Jelfs research group at Imperial
College London (<http://www.jelfs-group.org/>,
<https://github.com/JelfsMaterialsGroup>).

This code was reviewed and edited by: Lukas Turcani
(<https://github.com/lukasturcani>)

License
-------

This project is licensed under the MIT license.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _`First Paper Example`: first_paper_example.html
