SpinDry
=======

:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_
:documentation: https://spindry.readthedocs.io/en/latest/

A Monte Carlo-based host-guest conformer generator using cheap and unphysical
potentials.

Please submit an issue with any questions or bugs!

``SpinDry`` uses the Monte-Carlo/molecule interface provided by my other code
``MCHammer`` (https://github.com/andrewtarzia/MCHammer)

Installation
------------

Install using pip:

.. code-block:: bash

  pip install spindry


Developer Setup
---------------

To develop with ``spindry``, you can clone the repo and use
`just <https://github.com/casey/just>`_ and `uv <https://docs.astral.sh>`_
to setup the dev environment:

.. code-block:: bash

  just setup


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

1. Define a translation of the guest by a random unit vector and a random
   [-1, 1) step along that vector.
2. Define a rotation of the guest by a random [-1, 1) *
   ``rotation_step_size`` angle about a random unit axis.
3. Compute the system potential :math:`U_{nb}`, the nonbonded potential
   defined by the Lennard-Jones potential:

   .. math::

      U_{nb} = \sum_{i,j} \epsilon_{nb} \left[
          \left( \frac{\sigma}{r_{ij}} \right)^{12}
          - \left( \frac{\sigma}{r_{ij}} \right)^{6}
      \right]

   where :math:`\epsilon_{nb}` defines the strength of the potential,
   :math:`\sigma` defines the position where the potential becomes
   repulsive, and :math:`r_{ij}` is the pairwise distance between atoms
   :math:`i` and :math:`j`.
4. Accept or reject the move. Accept if :math:`U_i < U_{i-1}` or
   :math:`\exp(-\beta (U_i - U_{i-1})) > R`, where :math:`R` is a random
   number in [0, 1) and :math:`\beta = 1 / (k_B T)` is the inverse
   temperature. Reject otherwise.
5. If ``num_conformers`` is reached, stop.

Examples
--------

The workflow for a porous organic cage built using *stk*
(https://stk.readthedocs.io/) is shown in ``examples/`` for a single guest
and multiple guests.

The Spinner class yields a ``SupraMolecule`` conformer. Only conformers that
pass the MC conditions are yielded. The examples in ``examples`` show how to
access the structures of these conformers as ``.xyz`` files or `stk` molecules.

Contributors and Acknowledgements
---------------------------------

I developed this code as a post doc in the Jelfs research group at Imperial
College London (http://www.jelfs-group.org/,
https://github.com/JelfsMaterialsGroup).

This code was reviewed and edited by: Lukas Turcani
(https://github.com/lukasturcani)

License
-------

This project is licensed under the MIT license.
