.. Synax documentation master file, created by
   sphinx-quickstart on Tue Jul 30 15:06:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


synax documentation
===================

Welcome to **synax**'s documentation!

**synax** is a Python package designed to simulate Galactic synchrotron emission, covering both total and polarized intensity. **synax** leverages the capabilities of `JAX <https://jax.readthedocs.io/en/latest/>`_, providing features like automatic differentiation (AD) and support for multiple platforms (CPU, GPU, TPU, etc.). With access to gradients, **Synax** can seamlessly integrate into the broader **JAX** ecosystem, enabling the use of efficient inference algorithms such as Hamiltonian Monte Carlo and ADAM optimization.


.. figure:: figures/haslam_opt.jpg
   :alt: Optimized haslam map
   :align: center

   Haslam 408 MHz map can be reproduced by optimizing a 3D grid **B** field with **synax**!

**synax** intensity is currently designed for frequencies above ~ 408 MHz as it does not include free-free emission and absorption which is prominent in lower frequencies. This feature will be added in upcoming versions.

For polarizations, **synax** is recommended to use with frequencies higher than 1 Ghz as lower frequencies requires fine integration step size to resolve the change in polarization angle changes along sightlines. This would exceed the typical memory limit of GPUs.

.. admonition:: Where to start?
    :class: tip

    🖥 A good place to get started is with the :doc:`source/installation` and then the
    :doc:`nb/Simulation_lsa` guide. 

    💡 Then :doc:`nb/Integration` and :doc:`nb/Fields` shows more details on how we generate the necessary variables.

    📖 We demonstrate some examples about what can we do with AD in :doc:`nb/Sampling_lsa` and optimising :doc:`nb/Optimising_haslam`.

    🐛 If you find bugs, please head on over to the GitHub issues page.
    
    👈 Check out the sidebar to find the full table of contents.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :hidden:

   source/installation
   nb/Simulation_lsa

.. toctree::
   :maxdepth: 1
   :caption: Examples:
   :hidden:

   nb/Integration
   nb/Fields
   nb/Sampling_lsa
   nb/Optimising_haslam
   nb/ExtraGalactic
   

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation:

   source/api
   GitHub Repository <https://github.com/dkn16/synax>

Attribution & Citation
======================

Please cite the following if you find this code useful in your
research. The BibTeX entries for the paper is:

.. code-block:: none

      @ARTICLE{2024arXiv241001136D,
         author = {{Diao}, Kangning and {Li}, Zack and {Grumitt}, Richard D.~P. and {Mao}, Yi},
         title = "{$\texttt{synax}$: A Differentiable and GPU-accelerated Synchrotron Simulation Package}",
         journal = {arXiv e-prints},
      keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
            year = 2024,
         month = oct,
            eid = {arXiv:2410.01136},
         pages = {arXiv:2410.01136},
            doi = {10.48550/arXiv.2410.01136},
   archivePrefix = {arXiv},
         eprint = {2410.01136},
   primaryClass = {astro-ph.IM},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241001136D},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

    


Changelog
=========
**0.1.0 (2024/08/01)**

- First version