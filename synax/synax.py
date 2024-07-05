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



# obtaining integration locations
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def obtain_positions(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=256,epsilon:float=1e-7):
    
    nx = jnp.sin(theta)*jnp.cos(phi)
    ny = jnp.sin(theta)*jnp.sin(phi)
    nz = jnp.cos(theta)
    
    max_val = jnp.max(jnp.abs(jnp.array([nx/(x_length-obs_coord[0]*jnp.sign(nx)),ny/(y_length-obs_coord[1]*jnp.sign(ny)),nz/(z_length-obs_coord[2]*jnp.sign(nz))])))
    
    int_points,step = jnp.linspace(0,1,num_int_points,endpoint=False,retstep=True)
    
    int_points = int_points + step*0.5
    
    xs = nx/max_val*int_points
    ys = ny/max_val*int_points
    zs = nz/max_val*int_points
    
    dl = (xs[0]**2+ys[0]**2+zs[0]**2)**0.5*2
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]),dl,jnp.array([nx,ny,nz])

# obtaining integration locations
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def obtain_positions_hammurabi(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=4,y_length:float=4,z_length:float=4,num_int_points:int=256,epsilon:float=1e-7):
    
    nx = jnp.sin(theta)*jnp.cos(phi)
    ny = jnp.sin(theta)*jnp.sin(phi)
    nz = jnp.cos(theta)
    
    #max_val = jnp.max(jnp.abs(jnp.array([nx/(x_length-obs_coord[0]*jnp.sign(nx)),ny/(y_length-obs_coord[1]*jnp.sign(ny)),nz/(z_length-obs_coord[2]*jnp.sign(nz))])))
    
    int_points,step = jnp.linspace(0,1,num_int_points,endpoint=False,retstep=True)
    
    int_points = int_points + step*0.5
    
    xs = x_length*int_points*nx#+obs_coord[0]
    ys = y_length*int_points*ny#+obs_coord[1]
    zs = z_length*int_points*nz#+obs_coord[2]
    
    dl = (xs[0]**2+ys[0]**2+zs[0]**2)**0.5*2
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]),dl,jnp.array([nx,ny,nz])

@jax.jit
def C_page(x:float,y:float,z:float,C0:float = 1.,hr:float=5,hd:float=1):
    #x = x
    #z = z
    #c = (x**2+y**2+z**2)/jnp.max(jnp.array([x**2+y**2+z**2,9.]))+1e-7
    
    return C0*jnp.exp(-jnp.sqrt(x**2+y**2)/hr)/jnp.cosh(z/hd)**2#*(1-jnp.floor(c))


def C_uni(x:float,y:float,z:float,C0:float = 1.,):
    x = x+8.3
    z = z-0.006
    c = (x**2+y**2+z**2)/jnp.max(jnp.array([x**2+y**2+z**2,16.]))#+1e-7
    return 1-jnp.floor(c)

R_obs = (8.3**2+0.006**2)**0.5

def C_sun(x:float,y:float,z:float,C0:float = 6.4e1,):
    z = jnp.abs(z)
    factor1 = 1-jnp.floor(z/jnp.max(jnp.array([z,1])))
    R = (x**2+y**2)**0.5
    C = C0*jnp.exp(-(R-R_obs)/8-z)
    
    return factor1*C

C_earth = C_page(-8.3,0,0.006)
C_page_vmap = jax.vmap(lambda x,y,z:C_page(x,y,z))
C_uni_vmap = jax.vmap(lambda x,y,z:C_uni(x,y,z))
C_sun_vmap = jax.vmap(lambda x,y,z:C_sun(x,y,z))
