# synax/bfield.py

import jax,interpax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
from functools import partial
import scipy.constants as const
from typing import List, Tuple, Union,Dict

class B_jf12:
    """
        Initializer method (constructor)
        Args:
            instance_attribute: Attribute unique to each instance
        """
        # Instance attributes (unique to each instance)

    # Class attributes (shared by all instances)
    #class_attribute = 'I am a class attribute'

    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]]):
        
        get_index_vmap = jax.vmap(self.get_index)

        # Add it as an instance method
        setattr(self, 'get_index_vmap', get_index_vmap)
        
        self.rho = (coords[0]**2+coords[1]**2+coords[2]**2)**(1/2)
        self.r = (coords[0]**2+coords[1]**2)**(1/2)
        self.z = coords[2]
        self.shape = coords[2].shape
        self.phi = jnp.arctan2(coords[1],coords[0])
        self.indexs = self.get_index_vmap(self.r.reshape(-1),self.phi.reshape(-1),self.z.reshape(-1))
        
        
        B_calc_vmap = jax.vmap(self.B_calc,in_axes=(None, 0,0,0,0,0,0,0))
        setattr(self, 'B_calc_vmap', B_calc_vmap)
        
        self.Rmax = 20 #outer boundary of GMF
        self.rho_gc = 1. #interior boundary of GMF
        #self.inc = 11.5*jnp.pi/180. #inclination, in degrees

        self.rmin = 5. # outer boundary of the molecular ring region
        self.rcent = 3.# inner boundary of the molecular ring region (field is zero within this region)

        self.f = jnp.array([ 0.130, 0.165, 0.094, 0.122,
              0.13,  0.118, 0.084, 0.156]) # fractions of circumference spanned by each spiral arm

        self.rc_b = jnp.array([0,5.1,  6.3,  7.1, 8.3, 9.8,
              11.4, 12.7, 15.5]) # the radii where the spiral arm boundaries cross the negative x-axis
        
        ones_field = jnp.ones_like(coords[0])
        mask = (self.rho<self.rho_gc)|(self.r>self.Rmax)
        self.total_mask = ones_field.at[mask].set(1e-16)
        ones_field = jnp.ones_like(coords[0])
        mask = (self.r<self.rcent)
        self.rcent_mask = ones_field.at[mask].set(1e-16)
        ones_field = jnp.ones_like(coords[0])
        mask = (self.r<self.rmin)
        self.rmin_mask = ones_field.at[mask].set(1e-16)

    @staticmethod
    def B_calc(jf12_params: Dict[str,float],r:float,phi:float,z:float,is_r_less_rmin = 1.,is_r_less_rcent = 1.,b_disk = 1.,mask = 1.):
        #r = (x**2+y**2)**1/2
        #phi = jnp.arctan2(y,x)

        #disk components
        inc = 11.5*jnp.pi/180. #inclination, in degrees
    
        b0 = 5./r

        z_profile = 1/(1+jnp.exp(-2/jf12_params['w_disk']*(jnp.abs(z)-jf12_params['h_disk'])))

        B_cyl_disk = jnp.array([0,b0*jf12_params['b_ring']*(1-z_profile),0])*(1-is_r_less_rmin)

        B_cyl_disk += b_disk*is_r_less_rmin*jnp.array([jnp.sin(inc),jnp.cos(inc),0])*b0* (1 - z_profile)

        #toroidal components

        z_sign = jnp.sign(z)
        b1 = (z_sign+1.)/2*jf12_params['bn']+(1.-z_sign)/2*jf12_params['bs']
        rh = (z_sign+1.)/2*jf12_params['rn']+(1.-z_sign)/2*jf12_params['rs']

        bh = b1 * (1. - 1. / (1. + jnp.exp(-2. / jf12_params['wh'] * (r - rh)))) * jnp.exp(-jnp.abs(z)/ (jf12_params['z0']))
        B_cyl_h = jnp.array([0.,bh*z_profile,0.])

        #X-field

        rc_X = jf12_params['rpc_x'] + jnp.abs(z) / jnp.tan(jf12_params['x_theta'])
        rc_sign = jnp.sign(r-rc_X)
        rp_X = (r - jnp.abs(z)/jnp.tan(jf12_params['x_theta']))*(rc_sign+1.)/2 + (1- rc_sign)/2*r*jf12_params['rpc_x']/rc_X

        x_theta = jf12_params['x_theta']*(rc_sign+1.)/2+ (1- rc_sign)/2*jnp.arctan(jnp.abs(z)/(r-rp_X))

        B_X = jf12_params['b0_x']*rp_X/r*jnp.exp(-rp_X/jf12_params['r0_x'])*(rc_sign+1.)/2 + (1- rc_sign)/2*jf12_params['b0_x']*(jf12_params['rpc_x']/rc_X)**2*jnp.exp(-rp_X/jf12_params['r0_x'])

        B_cyl_X = jnp.array([B_X*jnp.cos(x_theta)*z_sign,0,B_X*jnp.sin(x_theta)])

        B_cyl = B_cyl_disk*is_r_less_rcent+B_cyl_h+B_cyl_X

        return jnp.array([B_cyl[0]*jnp.cos(phi) - B_cyl[1]*jnp.sin(phi),B_cyl[0]*jnp.sin(phi) + B_cyl[1]*jnp.cos(phi),B_cyl[2]])*mask
    
    @partial(jax.jit, static_argnums=(0,))
    def B_field(self,jf12_params):
        bv_b = jnp.array([jf12_params['b_arm_1'],jf12_params['b_arm_2'],jf12_params['b_arm_3'],jf12_params['b_arm_4'],jf12_params['b_arm_5'],jf12_params['b_arm_6'],jf12_params['b_arm_7'],0,0])
        b8 = -1*(self.f[:8]*bv_b[:8]).sum()/self.f[7]
        bv_b = bv_b.at[7].set(b8)

        disk_values = jnp.take(bv_b,self.indexs)
        B_field = self.B_calc_vmap(jf12_params,self.r.reshape(-1),self.phi.reshape(-1),self.z.reshape(-1),self.rmin_mask.reshape(-1),self.rcent_mask.reshape(-1),disk_values.reshape(-1),self.total_mask.reshape(-1)).reshape(self.shape + (3,))
        return B_field*1e-6
    
    
        
    
    @staticmethod
    def get_index(r:float,phi:float,z:float) -> int:
        rc_b = jnp.array([0,5.1,  6.3,  7.1, 8.3, 9.8,
              11.4, 12.7, 15.5])
        
        inc = 11.5*jnp.pi/180. #inclination, in degrees
        r_negx1 = r * jnp.exp((jnp.pi - phi) / jnp.tan(jnp.pi / 2 - inc))
        r_negx2 = r * jnp.exp((-1 * jnp.pi - phi) / jnp.tan(jnp.pi / 2 - inc))
        r_negx3 = r * jnp.exp((-3 * jnp.pi - phi) / jnp.tan(jnp.pi / 2 - inc))
        #r_negx4 = r * jnp.exp((-5 * jnp.pi - phi) / jnp.tan(jnp.pi / 2 - inc))

        r_negx = jnp.where(r_negx1 <= rc_b[8], r_negx1, 
                           jnp.where(r_negx2 <= rc_b[8], r_negx2, 
                                     r_negx3))

        index = jnp.searchsorted(rc_b, r_negx) 
        return index -1

    def __str__(self):
        """
        String representation of the instance
        """
        return f'B_jf12'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'B_jf12'


class B_lsa():
    
    class_attribute = 'I am a class attribute'

    def __init__(self, coords:Union[jax.Array,List[jax.Array],Tuple[jax.Array]]):
        
        self.rho = (coords[0]**2+coords[1]**2+coords[2]**2)**(1/2)
        self.r = (coords[0]**2+coords[1]**2)**(1/2)
        self.cos_p = coords[0]/self.r
        self.sin_p = coords[1]/self.r
        
        self.z = coords[2]
        self.shape = coords[2].shape
        
        ones_field = jnp.ones_like(coords[0])
        mask = (self.r<3)|(self.r>20.)
        self.total_mask = ones_field.at[mask].set(1e-16)
        B_calc_vmap = jax.vmap(self.B_calc,in_axes=(None, 0,0,0,0,0))
        setattr(self, 'B_calc_vmap', B_calc_vmap)
        
    @staticmethod
    def B_calc(lsa_params,r,z,cos_p,sin_p,mask):
        psi = lsa_params["psi0"]+lsa_params["psi1"]*jnp.log(r/8)
        chi = lsa_params["chi0"]*jnp.tanh(z)
        
        return jnp.array([jnp.sin(psi)*jnp.cos(chi)*cos_p - jnp.cos(psi)*jnp.cos(chi)*sin_p ,
                         jnp.sin(psi)*jnp.cos(chi)*sin_p + jnp.cos(psi)*jnp.cos(chi)*cos_p ,
                         jnp.sin(chi),])*mask*lsa_params["b0"]
        
    @partial(jax.jit, static_argnums=(0,))
    def B_field(self,lsa_params):
        
        return (self.B_calc_vmap(lsa_params,self.r.reshape(-1),self.z.reshape(-1),self.cos_p.reshape(-1),self.sin_p.reshape(-1),self.total_mask.reshape(-1))*1e-6).reshape(self.shape+  (3,))
    
    def __str__(self):
        """
        String representation of the instance
        """
        return f'B_lsa'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'B_lsa'


class B_grid():
    
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
    def B_field(self,B_field_grid):
        
        return self.field_calc(self.pos,B_field_grid).reshape(self.shape+(3,))
    
    def __str__(self):
        """
        String representation of the instance
        """
        return f'B_grid'

    def __repr__(self):
        """
        Official string representation of the instance
        """
        return f'B_grid'