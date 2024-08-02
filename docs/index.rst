.. Synax documentation master file, created by
   sphinx-quickstart on Tue Jul 30 15:06:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Synax documentation
===================

Welcome to Synax's documentation!

`synax` is a python package to simulate Galactic sychrotron emission, for both intensity and polarized intensity. 
`synax` is heavily based on `jax <https://jax.readthedocs.io/en/latest/installation.html/>`_ to obtain the features such as automatic differentiation (AD) and multi-platform (CPU, GPU, TPU, etc. ) support.
With the access to the gradient, one can integrate `synax` with the whole `jax` ecosystem easily, and perform efficient inferecen algorithms such as Hamitonian Monte Carlo and ADAM optimization.

.. figure:: figures/haslam_opt.pdf
   :alt: Optimized haslam map
   :align: center

   Haslam 408 MHz map can be reproduced by optimizing a 3D grid **B** field with `synax`!

.. admonition:: Where to start?
    :class: tip

    🖥 A good place to get started is with the :doc:`source/installation` and then the
    :doc:`nb/Simulation_lsa` guide. 

    💡 Then :doc:`nb/Integration` and :doc:`nb/Fields` shows more details on how we generate the necessary variables.

    📖 We demonstrate some examples about what can we do with AD in :doc:`nb/Sampling_lsa` and .

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
   

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation:

   source/api
   GitHub Repository <https://github.com/dkn16/synax>

Attribution & Citation
======================

Please cite the following if you find this code useful in your
research. The BibTeX entries for the papers are::

    To be specified later.

    


Changelog
=========
**0.1.0 (2024/08/01)**

- First version