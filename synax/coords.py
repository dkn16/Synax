
import jax.numpy as jnp
import jax
import numpy as np
import healpy as hp
from functools import partial
import scipy.constants as const
# obtaining integration locations
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def obtain_positions(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=512,epsilon:float=1e-7):
    
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
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]).T,dl,jnp.array([nx,ny,nz])

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
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]).T,dl,jnp.array([nx,ny,nz])

#obtain_vmap = jax.vmap(obtain_positions,in_axes=(0,0,None,None,None,None,None,None))

def get_healpix_positions(nside = 64,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=512,epsilon:float=1e-7):
    obtain_vmap = jax.vmap(lambda theta,phi:obtain_positions(theta,phi,obs_coord = obs_coord,x_length=x_length,y_length=y_length,z_length=z_length,num_int_points=num_int_points,epsilon=epsilon))
    n_pixs = np.arange(0,12*nside**2)
    theta,phi = hp.pix2ang(nside,n_pixs)
    poss,dls,nhats = obtain_vmap(theta,phi)
    return poss.transpose((2,0,1)),dls,nhats