
import jax.numpy as jnp
import jax,math
import numpy as np
import healpy as hp
from functools import partial
import scipy.constants as const
# obtaining integration locations
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def obtain_positions(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=512,epsilon:float=1e-7):
    """
    Calculate the integration points along one line of sight, from the location of the observer to the box boundary

    Args:
        theta (float): In unit of rad. The galactic longitude.
        phi (float): In unit of rad. The galactic co-lattitude. These values can be automatically generate by `healpy.pix2ang` with `lonlat = False`.
        obs_coord (tuple[float]): In unit of kpc. the location of observer.
        x_length (float): In unit of kpc. half of the box length along x-axis.
        y_length (float): In unit of kpc. half of the box length along y-axis.
        z_length (float): In unit of kpc. half of the box length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - pos (jnp.Array): In unit of kpc. 2D array of shape (num_int_points,3), coordinates of integration points along one sightline specified by (theta,phi).
            - dl (jnp.float): In unit of kpc. length of each integration segment.
            - nhat (jnp.Array): In unit of rad. 1D array of shape (3), unit vector of this LoS.
    """
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
    """
    Calculate the integration points along one line of sight in hammurabi way. Unlike integrate to the box boundary, now we integrate to a certain distance (x_length,y_length,z_length) way from observer.

    Args:
        theta (float): In unit of rad. The galactic longitude.
        phi (float): In unit of rad. The galactic co-lattitude. These values can be automatically generate by `healpy.pix2ang` with `lonlat = False`.
        obs_coord (tuple[float]): In unit of kpc. the location of observer.
        x_length (float): In unit of kpc. integration length along x-axis.
        y_length (float): In unit of kpc. integration length along y-axis.
        z_length (float): In unit of kpc. integration length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - pos (jnp.Array): In unit of kpc. 2D array of shape (num_int_points,3), coordinates of integration points along one sightline specified by (theta,phi).
            - dl (jnp.float): In unit of kpc. length of each integration segment.
            - nhat (jnp.Array): In unit of rad. 1D array of shape (3), unit vector of this LoS.
    """
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



def get_healpix_positions(nside = 64,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=512,epsilon:float=1e-7):
    """
    Calculate the integration points along each line of sight for a `HEALPix` map with given `nside`, from the location of the observer to the box boundary. A `HEALPix` map with a given `nside` should contains `npix = 12*nside**2` pixels.

    Args:
        nside (int): `NSIDE` of the `HEALPix` map.
        obs_coord (tuple[float]): In unit of kpc. the location of observer.
        x_length (float): In unit of kpc. half of the box length along x-axis.
        y_length (float): In unit of kpc. half of the box length along y-axis.
        z_length (float): In unit of kpc. half of the box length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - poss (jnp.Array): In unit of kpc. 3D array of shape (`npix`,`num_int_points`,3), coordinates of integration points along all sightlines of a `HEALPix` map.
            - dls (jnp.Array): In unit of kpc. 1D array of shape (`npix`), length of integration segment for all sightlines.
            - nhats (jnp.Array): In unit of rad. 2D array of shape (`npix`,3), unit vector of LoS for all pixels.
    """
    
    obtain_vmap = jax.vmap(lambda theta,phi:obtain_positions(theta,phi,obs_coord = obs_coord,x_length=x_length,y_length=y_length,z_length=z_length,num_int_points=num_int_points,epsilon=epsilon))
    n_pixs = np.arange(0,12*nside**2)
    theta,phi = hp.pix2ang(nside,n_pixs)
    poss,dls,nhats = obtain_vmap(theta,phi)
    return poss.transpose((2,0,1)),dls,nhats