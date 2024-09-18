**synax is a GPU accelerated automatic differentiable Galactic synchrotron simulation, powered by [JAX](https://jax.readthedocs.io)**

[![License: GPL v3](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/dkn16/synax/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/synax/badge/?version=latest)](https://synax.readthedocs.io/en/latest/?badge=latest)


# Getting started

## Brief introduction

Here's an edited version of the README file for your package:

---

# synax

**synax** is a Python package designed to simulate Galactic synchrotron emission, covering both total and polarized intensity. It leverages the power of JAX, offering features like automatic differentiation (AD) and multi-platform support (CPU, GPU, TPU). By providing gradient access, **synax** integrates smoothly into the JAX ecosystem, enabling the use of efficient inference algorithms like Hamiltonian Monte Carlo (HMC) and ADAM optimization.

![Haslam 408 MHz map can be reproduced by optimizing a 3D grid B field with synax](figures/haslam_opt.pdf)

**Intensity Simulation**: Currently, **synax** supports frequencies above ~408 MHz. The package does not yet include free-free emission and absorption, which are more significant at lower frequencies. These features will be added in future versions.

**Polarization Simulation**: For accurate polarization results, **synax** is recommended for frequencies above 1 GHz. Lower frequencies require finer integration step sizes to account for rapid changes in polarization angles along sightlines, which may exceed typical GPU memory limits.

--- 

Let me know if you'd like further revisions!

## Documentation

Read the docs at [synax.readthedocs.io](https://synax.readthedocs.io) for more information, examples and tutorials.

## Installation

we currently only supports install from source:

```bash
git clone https://github.com/dkn16/synax.git
cd synax
python setup.py install
```

pip channel will be constructed later.

## Basic example

For instance, if you wanted to calculate the Galactic synchrotron emission, you can carry out a 3-step procedure:

### Generating integration points
```python
import jax
import synax

nside = 64 # NSIDE of your desired sky map
num_int_points = 512 # integration points along line-of-sight

poss,dls,nhats = synax.coords.get_healpix_positions(nside=nside,num_int_points = num_int_points ) # Get the integration points
```

### Generating required fields
```python
lsa_params = {"b0":1.2,
               "psi0":27.0*np.pi/180,
               "psi1":0.9*np.pi/180,
               "chi0":25.0*np.pi/180}

B_generator = synax.bfield.B_lsa(poss) # define a LSA B-field model
B_field = B_generator.B_field(lsa_params) # generate B-field

C_field = ...# generate cosmic ray field

TE_field = ...# generate ISM thermal electron field

spectral_index = 3.#generate spectral index
```
### Simulate
```python
simer = synax.synax.Synax()#define the simer

freq = 2.4

sync = simer.sim(freq,B_field,C_field,TE_field,nhats,dls,spectral_index) # simulate!

sync['I'],sync['Q'],sync['U'] #I,Q,U maps, respectively
```


# Attribution & Citation

Please cite the following papers if you found this code useful in your research:

```bash
To be submitted soon
```

# Licence

Copyright 2022-Now Kangning Diao, Zack Li and Richard D.P. Grumitt.

``synax`` is free software made available under the MIT License. For details see the `LICENSE` file.