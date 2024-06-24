import jax.numpy as jnp
import jax
import numpy as np

import scipy.constants as const

q_converter = 1/(4*np.pi*const.epsilon_0)**0.5

B_converter = (4*np.pi/const.mu_0)**0.5

freq_irrelavent_const = (const.e*q_converter)**3/(const.electron_mass*const.speed_of_light**2)*(1/(np.pi*2))*(np.sqrt(3)/(8*np.pi))*1e19 # moves kpc = 1e19 m here.

elect_combi = 2/3*const.electron_mass*const.speed_of_light/(const.e*q_converter)

kpc = 3.08567758

@jax.jit
def sync_I_const(freq,spectral_index: float=3.):
    
    gamma_func_1 = jax.scipy.special.gamma(spectral_index/4.-1/12.)
    
    gamma_func_2_process = (2e-4*B_converter)**(spectral_index/2.+0.5)/(spectral_index+1)*jax.scipy.special.gamma(spectral_index/4+19/12.)# the transition from micro-Gauss to tesla is here.
    
    omega = 2*jnp.pi*freq*1e9
    
    consts = freq_irrelavent_const*(omega*elect_combi)**(0.5-spectral_index/2)*gamma_func_1*gamma_func_2_process
    
    return consts*kpc/(2*const.Boltzmann*freq**2*1e18/(const.speed_of_light**2))

@jax.jit
def sync_emiss_I(freq:float, b_perp: jax.Array,C:jax.Array,spectral_index: float=2.):
    # calculating the  synchrotron emissivity 
    # Input:
    #   freq (float): frequency to be computed.
    #   b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
    #   C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
    #   spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.
    # Output:
    #   j (jax.Array): parallel emissivity for the synchrotron emission.
    

    return b_perp**(0.5+spectral_index/2)*C*sync_I_const(freq,spectral_index=spectral_index)



# obtaining integration locations
@jax.jit
def obtain_positions(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.06),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=64,epsilon:float=1e-6):
    
    nx = jnp.sin(theta+epsilon)*jnp.cos(phi+epsilon)
    ny = jnp.sin(theta+epsilon)*jnp.sin(phi+epsilon)
    nz = jnp.cos(theta+epsilon)
    
    max_val = jnp.max(jnp.abs(jnp.array([nx/(x_length-obs_coord[0]*jnp.sign(nx)),ny/(y_length-obs_coord[1]*jnp.sign(ny)),nz/(z_length-obs_coord[2]*jnp.sign(nz))])))
    
    int_points = jnp.linspace(0,1,num_int_points)
    
    xs = nx/max_val*int_points
    ys = ny/max_val*int_points
    zs = nz/max_val*int_points
    
    dl = (xs[1]**2+ys[1]**2+zs[1]**2)**0.5
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]),dl

