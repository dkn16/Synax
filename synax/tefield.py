import os
import sys
import jax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp

import interpax
from functools import partial
from typing import List, Tuple, Union,Dict


class TE_grid():
    
    class_attribute = 'I am a class attribute'

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