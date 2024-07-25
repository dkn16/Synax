# synax/synax.py

import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
import scipy.constants as const

q_converter = 1/(4*np.pi*const.epsilon_0)**0.5

B_converter = (4*np.pi/const.mu_0)**0.5

freq_irrelavent_const = (const.e*q_converter)**3/(const.electron_mass*const.speed_of_light**2)*(np.sqrt(3)/(8*np.pi))*1e19 # moves kpc = 1e16 km here.

elect_combi = 2/3*const.electron_mass*const.speed_of_light/(const.e*q_converter)

kpc = 3.08567758

temp_covert = (const.hbar*1e9)/(const.Boltzmann*2.725)
#return _t*(np.exp(p)-1.)**2/(p**2*np.exp(p))

@jax.jit
def sync_I_const(freq,spectral_index: float=3.):
    
    gamma_func_1 = jax.scipy.special.gamma(spectral_index/4.-1/12.)
    
    gamma_func_2_process = (2e-4*B_converter)**(spectral_index/2.+0.5)/(spectral_index+1)*jax.scipy.special.gamma(spectral_index/4+19/12.)# the transition from micro-Gauss to tesla is here.
    
    omega = 2*jnp.pi*freq*1e9
    
    freq_irrelavent = freq_irrelavent_const/(2*const.Boltzmann*freq**2*1e18/(const.speed_of_light**2))
    
    consts = freq_irrelavent*(omega*elect_combi)**(0.5-spectral_index/2)*gamma_func_1*gamma_func_2_process
    
    p = freq*temp_covert
    
    return consts*kpc*(jnp.exp(p)-1.)**2/(p**2*jnp.exp(p))

@jax.jit
def sync_P_const(freq,spectral_index: float=3.):
    
    gamma_func_1 = jax.scipy.special.gamma(spectral_index/4.-1/12.)
    
    gamma_func_2_process = (2e-4*B_converter)**(spectral_index/2.+0.5)/(4.)*jax.scipy.special.gamma(spectral_index/4+7/12.)# the transition from micro-Gauss to tesla is here.
    
    omega = 2*jnp.pi*freq*1e9
    
    freq_irrelavent = freq_irrelavent_const/(2*const.Boltzmann*freq**2*1e18/(const.speed_of_light**2))
    
    consts = freq_irrelavent*(omega*elect_combi)**(0.5-spectral_index/2)*gamma_func_1*gamma_func_2_process
    
    p = freq*temp_covert
    
    return consts*kpc*(jnp.exp(p)-1.)**2/(p**2*jnp.exp(p))

@jax.jit
def sync_emiss_I(freq:float, b_perp: jax.Array,C:jax.Array,spectral_index: float=3.):
    # calculating the  synchrotron emissivity 
    # Input:
    #   freq (float): frequency to be computed.
    #   b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
    #   C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
    #   spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.
    # Output:
    #   j (jax.Array): parallel emissivity for the synchrotron emission.
    

    return b_perp**(0.5+spectral_index*0.5)*C*sync_I_const(freq,spectral_index=spectral_index)

@jax.jit
def sync_emiss_P(freq:float, b_perp: jax.Array,C:jax.Array,spectral_index: float=3.):
    # calculating the  synchrotron emissivity 
    # Input:
    #   freq (float): frequency to be computed.
    #   b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
    #   C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
    #   spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.
    # Output:
    #   j (jax.Array): parallel emissivity for the synchrotron emission.
    

    return b_perp**(0.5+spectral_index*0.5)*C*sync_P_const(freq,spectral_index=spectral_index)



