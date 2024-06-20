import jax.numpy as jnp
import jax

@jax.jit
def sync_I_const(freq,spectral_index: float=-2.51):
    return 1

@jax.jit
def sync_emiss_I(freq:float, b_perp: jax.Array,C:jax.Array,spectral_index: float=-2.51):
    # calculating the  synchrotron emissivity 
    # Input:
    #   freq (float): frequency to be computed.
    #   b_perp (jax.Array): 3D magnetic field ($B_t$) perpendicular to the LOS.
    #   C (jax.Array): 3D field, defined by $N(\gamma)d\gamma = C\gamma^{-p}d\gamma$. Varied at different locations.
    #   spectral_index (float or jax.Array): spectrum of cosmic ray electron spectrum.
    # Output:
    #   j (jax.Array): parallel emissivity for the synchrotron emission.
    

    return b_perp**(0.5-spectral_index/2)*C*sync_I_const(freq,spectral_index=spectral_index)



# obtaining integration locations
@jax.jit
def obtain_positions(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.06),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=256,epsilon:float=1e-6):
    
    nx = jnp.sin(theta+epsilon)*jnp.cos(phi+epsilon)
    ny = jnp.sin(theta+epsilon)*jnp.sin(phi+epsilon)
    nz = jnp.cos(theta+epsilon)
    
    max_val = jnp.max(jnp.abs(jnp.array([nx/(x_length-obs_coord[0]*jnp.sign(nx)),ny/(y_length-obs_coord[1]*jnp.sign(ny)),nz/(z_length-obs_coord[2]*jnp.sign(nz))])))
    
    int_points = jnp.linspace(0,1,256)
    
    xs = nx/max_val*int_points+obs_coord[0]
    ys = ny/max_val*int_points+obs_coord[1]
    zs = nz/max_val*int_points+obs_coord[2]
    
    return jnp.array([xs,ys,zs])

