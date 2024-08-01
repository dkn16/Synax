import os
import sys
import jax,interpax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp


from functools import partial
from typing import List, Tuple, Union,Dict


class C_WMAP():
    
    """
    WMAP C field model. (https://iopscience.iop.org/article/10.1086/513699)

    Args:
        coords (Union[jax.Array,List[jax.Array],Tuple[jax.Array]]): coordinates of all integration points. Should be of size (3,...), for example ``coords[0]`` is the x-coordinates.

    Returns:
        A instance of WMAP C field generator.
    """
    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]]):
        
        self.r = (coords[0]**2+coords[1]**2)**(1/2)
        
        self.z = coords[2]
        self.shape = coords[2].shape
        
        
        C_calc_vmap = jax.vmap(self.C_calc,in_axes=(None, 0,0))
        setattr(self, 'C_calc_vmap', C_calc_vmap)
        
    @staticmethod
    def C_calc(WMAP_params, r:float,z:float):
        """
        Calculate wmap C-field at a given position ``(r,phi,z)``.

        Args:
            r (float): r in cylindrical coordinates.
            z (float): z in cylindrical coordinates.
        
        Returns:
            cosmic ray electron spectrum constant C in this position.
        """
        return WMAP_params['C0']*jnp.exp(-r/WMAP_params['hr'])/jnp.cosh(z/WMAP_params['hd'])**2#*(1-jnp.floor(c))
        
    @partial(jax.jit, static_argnums=(0,))
    def C_field(self,WMAP_params = {'C0':211.13068378473076,'hr':5.,'hd':1.}):
        """
        Calculate WMAP C-field at all positions specified by ``coords``.

        Args:
            WMAP_params (Dict[str,float]): A dict contains all parameters of the WMAP model.

        Returns:
            jnp.Array of shape (``coords[0].shape``). ``coords`` is the parameter of your C_WMAP instance.
        """
        return self.C_calc_vmap(WMAP_params,self.r.reshape(-1),self.z.reshape(-1)).reshape(self.shape)
    
    def __str__(self):
        """
        String representation of the instance
        """
        return f'C_WMAP'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'C_WMAP'
    
class C_uni():
    
    """
    Uniform C field model cenntered at ``center``

    Args:
        coords (Union[jax.Array,List[jax.Array],Tuple[jax.Array]]): coordinates of all integration points. Should be of size (3,...), for example ``coords[0]`` is the x-coordinates.
        center: center of uniform C field. default to be the earth.

    Returns:
        A instance of uniform C field generator.
    """

    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]],center = (-8.3,0.0,0.006)):
        
        self.x = coords[0] - center[0]
        self.y = coords[1] - center[1]
        self.z = coords[2] - center[2]
        self.shape = coords[2].shape
        
        C_calc_vmap = jax.vmap(self.C_calc,in_axes=(None, 0,0,0))
        setattr(self, 'C_calc_vmap', C_calc_vmap)
        
    @staticmethod
    def C_calc(Uni_params,x:float,y:float,z:float):
        """
        Calculate uniform C-field at a given position ``(x,y,z)``.

        Args:
            x (float): x in cartisian coordinates.
            y (float): y in cartisian coordinates.
            z (float): z in cartisian coordinates.
        
        Returns:
            cosmic ray electron spectrum constant C in this position.
        """
        c = (x**2+y**2+z**2)/jnp.max(jnp.array([x**2+y**2+z**2,Uni_params['rho0']**2]))#+1e-7
        return (1-jnp.floor(c))*Uni_params['C0']
    
    @partial(jax.jit, static_argnums=(0,))
    def C_field(self,Uni_params = {'C0':1.0,'rho0':4.,}):
        """
        Calculate uniform C-field at all positions specified by ``coords``.

        Args:
            Uni_params (Dict[str,float]): A dict contains all parameters of the uniform model.

        Returns:
            jnp.Array of shape (``coords[0].shape``). ``coords`` is the parameter of your C_uni instance.
        """
        return self.C_calc_vmap(Uni_params,self.x.reshape(-1),self.y.reshape(-1),self.z.reshape(-1)).reshape(self.shape)
    
    def __str__(self):
        """
        String representation of the instance
        """
        return f'C_uni'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'C_uni'
    
class C_grid():
    
    """
    grid C field model. See Synax paper for more details.

    Args:
        coords (Union[jax.Array,List[jax.Array],Tuple[jax.Array]]): coordinates of all integration points. Should be of size (3,...), for example ``coords[0]`` is the x-coordinates.
        coords_field (Union[jax.Array,List[jax.Array],Tuple[jax.Array]]): coords[i] is the 1D vector of coordinates along i-th axis. Since the grid is a regular 3D grid, 1D vectors are sufficient to represents the coordinates.

    Returns:
        A instance of grid C field generator.
    """

    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]],coords_field:Union[jax.Array,List[jax.Array],Tuple[jax.Array]]):
        
        self.x = coords[0] 
        self.y = coords[1] 
        self.z = coords[2]
        self.pos = coords

        self.xf = coords_field[0] 
        self.yf = coords_field[1] 
        self.zf = coords_field[2]

        self.shape = coords[2].shape
        
        field_calc = lambda pos,field: interpax.interp3d(pos[0].reshape(-1),pos[1].reshape(-1),pos[2].reshape(-1),self.xf,self.yf,self.zf,field,method='linear',extrap=True)
        setattr(self, 'field_calc', field_calc)

    
    @partial(jax.jit, static_argnums=(0,))
    def C_field(self,C_field_grid):
        """
        Calculate grid C-field at all positions specified by ``coords``.

        Args:
            C_field_grid (Dict[str,float]): your field in a regular 3D grid.

        Returns:
            jnp.Array of shape (``coords[0].shape``). ``coords`` is the parameter of your C_grid instance.
        """
        return self.field_calc(self.pos,C_field_grid).reshape(self.shape+(3,))
    
    def __str__(self):
        """
        String representation of the instance
        """
        return f'C_grid'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'C_grid'