import os
import sys
import jax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp


from functools import partial
from typing import List, Tuple, Union,Dict


class C_WMAP():
    
    class_attribute = 'I am a class attribute'

    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]]):
        
        self.r = (coords[0]**2+coords[1]**2)**(1/2)
        
        self.z = coords[2]
        self.shape = coords[2].shape
        
        
        C_calc_vmap = jax.vmap(self.C_calc,in_axes=(None, 0,0))
        setattr(self, 'C_calc_vmap', C_calc_vmap)
        
    @staticmethod
    def C_calc(WMAP_params, r:float,z:float):
    
        return WMAP_params['C0']*jnp.exp(-r/WMAP_params['hr'])/jnp.cosh(z/WMAP_params['hd'])**2#*(1-jnp.floor(c))
        
    @partial(jax.jit, static_argnums=(0,))
    def C_field(self,WMAP_params = {'C0':211.13068378473076,'hr':5.,'hd':1.}):
        
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
    
    class_attribute = 'I am a class attribute'

    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]],center = (-8.3,0.0,0.006)):
        
        self.x = coords[0] - center[0]
        self.y = coords[1] - center[1]
        self.z = coords[2] - center[2]
        self.shape = coords[2].shape
        
        C_calc_vmap = jax.vmap(self.C_calc,in_axes=(None, 0,0,0))
        setattr(self, 'C_calc_vmap', C_calc_vmap)
        
    @staticmethod
    def C_calc(Uni_params,x:float,y:float,z:float):
        
        c = (x**2+y**2+z**2)/jnp.max(jnp.array([x**2+y**2+z**2,Uni_params['rho0']**2]))#+1e-7
        return (1-jnp.floor(c))*Uni_params['C0']
    
    @partial(jax.jit, static_argnums=(0,))
    def C_field(self,Uni_params = {'C0':1.0,'rho0':4.,}):
        
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