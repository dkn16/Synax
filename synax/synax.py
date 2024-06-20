import jax.numpy as jnp
import jax

@jax.jit
def j_perp(b_perp: jax.Array,C:jax.Array,spectral_index: float=-2.51):
    # calculating the perpendicular synchrotron emissivity 
    # Input:
    #   b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
    #   C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
    #   spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.
    # Output:
    #   j_para (jax.Array): parallel emissivity for the synchrotron emission.


    return b_perp**spectral_index*C

@jax.jit
def j_para(b_perp: jax.Array,C:jax.Array,spectral_index: float=-2.51):
    # calculating the parallel synchrotron emissivity 
    # Input:
    #   b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
    #   C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
    #   spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.
    # Output:
    #   j_para (jax.Array): parallel emissivity for the synchrotron emission.

    return b_perp**spectral_index*C


def npixandweight(nside:int = 256,obs: tuple[float]=(0.1,0.1,0.1)):
    #For a given healpix map, assign the 3D field voxels and corresponding weight to each sightline
    return 0