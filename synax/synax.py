# synax/synax.py

import jax.numpy as jnp
import jax
import numpy as np

from functools import partial
import scipy.constants as const
from typing import List, Tuple, Union,Dict
jax.config.update("jax_enable_x64", True)

q_converter = 1/(4*np.pi*const.epsilon_0)**0.5

B_converter = (4*np.pi/const.mu_0)**0.5

freq_irrelavent_const = (const.e*q_converter)**3/(const.electron_mass*const.speed_of_light**2)*(np.sqrt(3)/(8*np.pi))*1e19 # moves kpc = 1e16 km here.

elect_combi = 2/3*const.electron_mass*const.speed_of_light/(const.e*q_converter)

kpc = 3.08567758

temp_covert = (const.hbar*1e9)/(const.Boltzmann*2.725)

rm_freq_irrelavent_const = (const.e*q_converter)**3/(const.electron_mass**2*const.speed_of_light**4)/(2*np.pi)*1e6*1e-4*B_converter*1e19*3.08 # moves 1/cm^3 = 1e6 1/m^3 1 gauss = 1e-4 tesla 1 kpc = 3.08e19 m here.

#return _t*(np.exp(p)-1.)**2/(p**2*np.exp(p))

@jax.jit
def sync_I_const(freq,spectral_index: float=3.):
    """
    calculating the constant irrelavent to b_perp and C in the synchrotron emissivity.

    Args:
       freq (float): frequency to be computed. In GHz.
       spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.

    Returns:
       (jax.Array): parallel emissivity constant for the synchrotron emission.
    """
    
    gamma_func_1 = jax.scipy.special.gamma(spectral_index/4.-1/12.)
    
    gamma_func_2_process = (2e-4*B_converter)**(spectral_index/2.+0.5)/(spectral_index+1)*jax.scipy.special.gamma(spectral_index/4+19/12.)# the transition from micro-Gauss to tesla is here.
    
    omega = 2*jnp.pi*freq*1e9
    
    freq_irrelavent = freq_irrelavent_const/(2*const.Boltzmann*freq**2*1e18/(const.speed_of_light**2))
    
    consts = freq_irrelavent*(omega*elect_combi)**(0.5-spectral_index/2)*gamma_func_1*gamma_func_2_process
    
    #p = freq*temp_covert
    
    return consts*kpc#*(jnp.exp(p)-1.)**2/(p**2*jnp.exp(p))

@jax.jit
def sync_P_const(freq,spectral_index: float=3.):
    """
    calculating the constant irrelavent to b_perp and C in the polarized synchrotron emissivity.

    Args:
       freq (float): frequency to be computed. In GHz.
       spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.

    Returns:
       (jax.Array): perpenndicular emissivity constant for the polarized synchrotron emission.
    """
    
    gamma_func_1 = jax.scipy.special.gamma(spectral_index/4.-1/12.)
    
    gamma_func_2_process = (2e-4*B_converter)**(spectral_index/2.+0.5)/(4.)*jax.scipy.special.gamma(spectral_index/4+7/12.)# the transition from micro-Gauss to tesla is here.
    
    omega = 2*jnp.pi*freq*1e9
    
    freq_irrelavent = freq_irrelavent_const/(2*const.Boltzmann*freq**2*1e18/(const.speed_of_light**2))
    
    consts = freq_irrelavent*(omega*elect_combi)**(0.5-spectral_index/2)*gamma_func_1*gamma_func_2_process
    
    #p = freq*temp_covert
    
    return consts*kpc#*(jnp.exp(p)-1.)**2/(p**2*jnp.exp(p))

@jax.jit
def sync_emiss_I(freq:float, b_perp: jax.Array,C:jax.Array,spectral_index: float=3.):
    """
    Calculating the  synchrotron emissivity.

    Args:
       freq (float): frequency to be computed. In GHz.
       b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
       C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
       spectral_index (float): spectrum of cosmic ray electron spectrum.

    Returns:
       (jax.Array): parallel emissivity for the synchrotron emission.
    """
    

    return b_perp**(0.5+spectral_index*0.5)*C*sync_I_const(freq,spectral_index=spectral_index)

@jax.jit
def sync_emiss_P(freq:float, b_perp: jax.Array,C:jax.Array,spectral_index: float=3.):
    """
    Calculating the polarized synchrotron emissivity.

    Args:
       freq (float): frequency to be computed. In GHz.
       b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
       C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
       spectral_index (float): spectrum of cosmic ray electron spectrum.
    Returns:
       (jax.Array): perpendicular emissivity for the polarized synchrotron emission.
    """
    

    return b_perp**(0.5+spectral_index*0.5)*C*sync_P_const(freq,spectral_index=spectral_index)



class Synax():
    
    """
    Synax simulator
    Args:
        sim_I (bool): whether sim synchrotron intensity.
        sim_P (bool): whether sim polarized synchrotron intensity.


    Returns:
        A instance of Synax simulator.
    """

    def __init__(self, sim_I = True,sim_P = True):
        
        self.sim_I = sim_I
        self.sim_P = sim_P
        
    @staticmethod
    @jax.jit
    def RM(freq,B_field,TE_field,nhats,dls,B_los):
        """
        Calculate rotation measure.

        Args:
            freq (float): frequency to be computed. In GHz.
            B_field (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS at different places.
            TE_field (jax.Array): 3D electron density field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
            nhats (jnp.Array): In unit of rad. unit vector of different LoS.
            dls (jnp.float): In unit of kpc. length of each integration segment for every LoS.
            B_los (jax.Array): LoS B-field magnitude.

            

        Returns:
            tuple:
                - fd (jnp.Array): rotation measure for each LoS integration point.
                - fd_q (jnp.Array): cos(2*polarized angle), for Q map calculation.
                - fd_u (jnp.Array):  sin(2*polarized angle), for U map calculation.
        """
        phis = rm_freq_irrelavent_const*TE_field*B_los
        sinb = nhats[...,2]
        cosb = jnp.sqrt(1-sinb**2)
        cosl = nhats[...,0]/cosb
        sinl = nhats[...,1]/cosb

        Bz = B_field[...,2]
        By = B_field[...,1]
        Bx = B_field[...,0]
        tanchi0 = (Bz*cosb[:,jnp.newaxis]-sinb[:,jnp.newaxis]*(cosl[:,jnp.newaxis]*Bx+By*sinl[:,jnp.newaxis]))/(Bx*sinl[:,jnp.newaxis]-By*cosl[:,jnp.newaxis]+1e-16)
        chi0 = jnp.arctan(tanchi0)
        phi_int = jnp.cumsum(phis,axis=1)*dls[:,jnp.newaxis]

        fd = phi_int*const.c**2/(freq**2*1e18)
        fd_q = jnp.cos(2*fd+2*chi0)
        fd_u = jnp.sin(2*fd+2*chi0)
        return fd,fd_q,fd_u
    
    @staticmethod
    @jax.jit
    def B_los(B_field,nhats):
        """
        Calculate LoS B-field.

        Args:
            B_field (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS at different places.
            nhats (jnp.Array): In unit of rad. unit vector of different LoS.
            

        Returns:
            (jnp.Array): LoS B-field magnitude.
        """
        return -1*((nhats[:,jnp.newaxis,:]*B_field)).sum(axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def sim(self,freq,B_field,C_field,TE_field,nhats,dls,spectral_index):
        """
        Calculate sychrotron map.

        Args:
            freq (float): frequency to be computed. In GHz.
            B_field (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS at different places.
            C_field (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
            TE_field (jax.Array): 3D electron density field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
            nhats (jnp.Array): In unit of rad. unit vector of different LoS.
            dls (jnp.float): In unit of kpc. length of each integration segment for every LoS.
            spectral_index (float): spectrum of cosmic ray electron spectrum.

            

        Returns:
            dict:
               - dict['I'](jnp.Array): Sychrotron I map. return 0 if ``sim_I=False``
               - dict['Q'](jnp.Array): Sychrotron Q map. return 0 if ``sim_P=False``
               - dict['U'](jnp.Array): Sychrotron U map. return 0 if ``sim_P=False``
        """
        B_los = self.B_los(B_field,nhats)
        B_trans = ((B_field**2).sum(axis=-1)-B_los**2)**0.5
        Sync_I = 0.
        Sync_Q = 0.
        Sync_U = 0.

        if self.sim_I:
            emiss = sync_emiss_I(freq,B_trans,C_field,spectral_index=spectral_index)
            Sync_I = emiss.sum(axis=-1)*dls

        if self.sim_P:
            fd,fd_q,fd_u = self.RM(freq,B_field,TE_field,nhats,dls,B_los)
            emiss = sync_emiss_P(freq,B_trans,C_field,spectral_index=spectral_index)
            Sync_Q = (emiss*fd_q).sum(axis=-1)*dls
            Sync_U = (emiss*fd_u).sum(axis=-1)*dls
        return {'I':Sync_I,'Q':Sync_Q,'U':Sync_U}

    def __str__(self):
        """
        String representation of the instance
        """
        return f'Synax'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'Synax'
