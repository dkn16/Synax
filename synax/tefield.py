import os
import sys
import jax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp

import interpax
from functools import partial
from typing import List, Tuple, Union,Dict


class TE_grid():
    
    """
    grid TE field model. See Synax paper for more details.

    Args:
        coords (Union[jax.Array,List[jax.Array],Tuple[jax.Array]]): coordinates of all integration points. Should be of size (3,...), for example ``coords[0]`` is the x-coordinates.
        coords_field (Union[jax.Array,List[jax.Array],Tuple[jax.Array]]): coords[i] is the 1D vector of coordinates along i-th axis. Since the grid is a regular 3D grid, 1D vectors are sufficient to represents the coordinates.

    Returns:
        A instance of grid TE field generator.
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
    def TE_field(self,te_field_grid):
        """
        Calculate grid TE-field at all positions specified by ``coords``.

        Args:
            TE_field_grid (Dict[str,float]): your field in a regular 3D grid.

        Returns:
            jnp.Array of shape (``coords[0].shape``). ``coords`` is the parameter of your TE_grid instance.
        """
        return self.field_calc(self.pos,te_field_grid).reshape(self.shape)
    
    def __str__(self):
        """
        String representation of the instance
        """
        return f'TE_grid'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'TE_grid'