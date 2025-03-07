Installation Guide
==================

Dependencies
------------

**synax** depends on ``jax``, ``numpy``, ``scipy``, ``healpy``, ``interpax``. We strongly recommend install ``jax`` manually following `their documentation <https://jax.readthedocs.io/en/latest/installation.html/>`_ first, so that you can install the version that fits your platform.

Specifically, you can install the dependencies using the following command:

.. code-block:: bash
      
      pip install numpy scipy healpy
      pip install 'jax[cpu]==0.4.34'  # or 'jax[cuda12]' if you have a  NVidia GPU
      pip install interpax

There're currently compatibility issues with ``jax`` and ``interpax``. We recommend using ``jax==0.4.43``.

From PyPi
------------------
We havn't upload our package to PyPi yet. Will do later.

From Source
------------------

To install the ``synax`` package from , follow these steps:

1. Open your terminal.
2. Navigate to your desired install directory.
3. Run the following command:

   .. code-block:: bash
      
      git clone https://github.com/dkn16/Synax.git
      cd Synax
      pip install .

This command will automatically clone the repo from github and install ``synax``. Users from Mainland China might find it easier to run ``git clone git@github.com:dkn16/Synax.git``.