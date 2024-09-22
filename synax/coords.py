
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
        phi (float): In unit of rad. The galactic co-lattitude. These values can be automatically generate by ``healpy.pix2ang`` with ``lonlat = False``.
        obs_coord (tuple[float]): In unit of kpc. the location of observer.
        x_length (float): In unit of kpc. half of the box length along x-axis.
        y_length (float): In unit of kpc. half of the box length along y-axis.
        z_length (float): In unit of kpc. half of the box length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - pos (jnp.Array): In unit of kpc. 2D array of shape (``num_int_points``,3), coordinates of integration points along one sightline specified by (theta,phi).
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
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]),dl,jnp.array([nx,ny,nz])

# obtaining integration locations
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def obtain_positions_hammurabi(theta,phi,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=4,y_length:float=4,z_length:float=4,num_int_points:int=256,epsilon:float=1e-7):
    """
    Calculate the integration points along one line of sight in hammurabi way. Unlike integrate to the box boundary, now we integrate to a certain distance ``(x_length,y_length,z_length)`` way from observer.

    Args:
        theta (float): In unit of rad. The galactic longitude.
        phi (float): In unit of rad. The galactic co-lattitude. These values can be automatically generate by ``healpy.pix2ang`` with ``lonlat = False``.
        obs_coord (tuple[float]): In unit of kpc. the location of observer.
        x_length (float): In unit of kpc. integration length along x-axis.
        y_length (float): In unit of kpc. integration length along y-axis.
        z_length (float): In unit of kpc. integration length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - pos (jnp.Array): In unit of kpc. 2D array of shape (``num_int_points``,3), coordinates of integration points along one sightline specified by ``(theta,phi)``.
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
    
    return jnp.array([xs+obs_coord[0],ys+obs_coord[1],zs+obs_coord[2]]),dl,jnp.array([nx,ny,nz])



def get_healpix_positions(nside = 64,obs_coord:tuple[float] = (-8.3,0.,0.006),x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=512,epsilon:float=1e-7):
    """
    Calculate the integration points along each line of sight for a ``HEALPix`` map with given `nside`, from the location of the observer to the box boundary. A ``HEALPix`` map with a given ``nside`` should contains ``npix = 12*nside**2`` pixels.

    Args:
        nside (int): ``NSIDE`` of the ``HEALPix`` map.
        obs_coord (tuple[float]): In unit of kpc. the location of observer.
        x_length (float): In unit of kpc. half of the box length along x-axis.
        y_length (float): In unit of kpc. half of the box length along y-axis.
        z_length (float): In unit of kpc. half of the box length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - poss (jnp.Array): In unit of kpc. 3D array of shape (``npix, num_int_points``, 3), coordinates of integration points along all sightlines of a ``HEALPix`` map.
            - dls (jnp.Array): In unit of kpc. 1D array of shape (``npix``), length of integration segment for all sightlines.
            - nhats (jnp.Array): In unit of rad. 2D array of shape (``npix``,3), unit vector of LoS for all pixels.
    """
    
    obtain_vmap = jax.vmap(lambda theta,phi:obtain_positions(theta,phi,obs_coord = obs_coord,x_length=x_length,y_length=y_length,z_length=z_length,num_int_points=num_int_points,epsilon=epsilon))
    n_pixs = np.arange(0,12*nside**2)
    theta,phi = hp.pix2ang(nside,n_pixs)
    poss,dls,nhats = obtain_vmap(theta,phi)
    return poss.transpose((1,0,2)),dls,nhats


def get_rotated_box_vertices(x, y, z, theta, phi):
    """
    Calculate the coordinates of the vertices of a rectangular box after rotation.

    Args:
        x, y, z: Half dimensions of the box along the X, Y, and Z axes. e.g. (-x,x) is the boundary of the box in x-axis
        theta: Rotation angle around the Z-axis (in radians).
        phi: Rotation angle around the Y-axis (in radians).

    Returns:
        - vertices_rotated: A (8, 3) array of the rotated vertices.
    """
    # Half dimensions
    hx, hy, hz = x , y , z 

    # Define the 8 vertices of the box
    signs = jnp.array([[1,  1,  1],
                       [1,  1, -1],
                       [1, -1,  1],
                       [1, -1, -1],
                       [-1, 1,  1],
                       [-1, 1, -1],
                       [-1,-1,  1],
                       [-1,-1, -1]])
    vertices = signs * jnp.array([hx, hy, hz])

    # Rotation matrices
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    Rz = jnp.array([[cos_theta, -sin_theta, 0],
                    [sin_theta,  cos_theta, 0],
                    [0,          0,         1]])

    Ry = jnp.array([[ cos_phi, 0, sin_phi],
                    [ 0,       1, 0      ],
                    [-sin_phi, 0, cos_phi]])

    # Combined rotation matrix
    R = Ry @ Rz

    # Rotate vertices
    vertices_rotated = vertices @ R.T

    return vertices_rotated

@jax.jit
def find_min_rectangle_xy(vertices_rotated):
    """
    Find the minimum axis-aligned rectangle in the X-Y plane that contains all projected vertices.

    Args:
        vertices_rotated: A (8, 3) array of the rotated vertices.

    Returns:
        tuple:
            - min_x,max_x,min_y,max_y
    """
    # Project onto X-Y plane
    xy_coords = vertices_rotated[:, :2]  # Take only X and Y coordinates

    # Find min and max X and Y
    min_x = jnp.min(xy_coords[:, 0])
    max_x = jnp.max(xy_coords[:, 0])
    min_y = jnp.min(xy_coords[:, 1])
    max_y = jnp.max(xy_coords[:, 1])



    return min_x,max_x,min_y,max_y


def vertical_line_intersects_box(vertices_rotated, x0, y0):
    """
    Determines whether the vertical line at (x0, y0) intersects the rotated box.

    Args:
        vertices_rotated: A (8, 3) array of the rotated vertices.
        x0, y0: Coordinates of the point(s) in the XY plane. Can be scalars or arrays.

    Returns:
        tuple:
            - intersects: Boolean array indicating whether each line intersects the box.
            - min_z: Array of minimum Z-values where the line intersects the box.
            - max_z: Array of maximum Z-values where the line intersects the box.
    """
    # Ensure x0 and y0 are arrays for broadcasting
    x0 = jnp.atleast_1d(x0)
    y0 = jnp.atleast_1d(y0)

    # Define the faces of the box using vertex indices
    faces = jnp.array([
        [0, 2, 6, 4],  # Top face
        [1, 3, 7, 5],  # Bottom face
        [0, 1, 3, 2],  # Side face
        [4, 5, 7, 6],  # Side face
        [0, 1, 5, 4],  # Side face
        [2, 3, 7, 6],  # Side face
    ], dtype=int)

    # For each face, define its two triangles
    triangles = jnp.concatenate([
        faces[:, [0, 1, 2]],
        faces[:, [0, 2, 3]],
    ], axis=0)  # Shape: (12, 3)

    # Get the vertices for each triangle
    p0 = vertices_rotated[triangles[:, 0]]  # Shape: (12, 3)
    p1 = vertices_rotated[triangles[:, 1]]
    p2 = vertices_rotated[triangles[:, 2]]

    # Compute plane normals
    v0 = p1 - p0  # Shape: (12, 3)
    v1 = p2 - p0
    normals = jnp.cross(v0, v1)  # Shape: (12, 3)
    A = normals[:, 0]
    B = normals[:, 1]
    C = normals[:, 2]

    # Compute D = -dot(normal, p0)
    D = -jnp.einsum('ij,ij->i', normals, p0)  # Shape: (12,)

    # Avoid division by zero for planes where C is zero
    C_nonzero = jnp.abs(C) > 1e-8

    # Prepare x0, y0 for broadcasting
    x0_b = x0[:, None]  # Shape: (N_points, 1)
    y0_b = y0[:, None]  # Shape: (N_points, 1)

    # Compute z at (x0, y0)
    numerator = -(A * x0_b + B * y0_b + D)  # Shape: (N_points, 12)
    denominator = C  # Shape: (12,)
    z = jnp.where(C_nonzero, numerator / C, jnp.nan)  # Shape: (N_points, 12)

    # Create point p
    x0_rep = x0_b * jnp.ones_like(z)
    y0_rep = y0_b * jnp.ones_like(z)
    p = jnp.stack([x0_rep, y0_rep, z], axis=-1)  # Shape: (N_points, 12, 3)

    # Compute vectors for barycentric coordinates
    v2 = p - p0  # Broadcasting over points

    # Compute dot products
    dot00 = jnp.einsum('ij,ij->i', v0, v0)  # Shape: (12,)
    dot01 = jnp.einsum('ij,ij->i', v0, v1)
    dot11 = jnp.einsum('ij,ij->i', v1, v1)

    # Corrected einsum indices
    dot02 = jnp.einsum('nij,ij->ni', v2, v0)  # Shape: (N_points, 12)
    dot12 = jnp.einsum('nij,ij->ni', v2, v1)

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01  # Shape: (12,)
    denom_nonzero = jnp.abs(denom) > 1e-8

    # Prepare denom for broadcasting
    denom = jnp.where(denom_nonzero, denom, jnp.nan)  # Shape: (12,)

    u = (dot11 * dot02 - dot01 * dot12) / denom  # Shape: (N_points, 12)
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1 - u - v

    # Check if point is inside triangle
    cond = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8) & \
           (u <= 1 + 1e-8) & (v <= 1 + 1e-8) & (u + v <= 1 + 1e-8)
    valid = cond & C_nonzero[None, :] & denom_nonzero[None, :]

    # Extract valid z values
    z_values = jnp.where(valid, z, jnp.nan)  # Shape: (N_points, 12)

    # Determine if any intersections occur
    intersects = jnp.any(valid, axis=1)  # Shape: (N_points,)

    # Compute min and max z-values for each point
    min_z = jnp.nanmin(z_values, axis=1)  # Shape: (N_points,)
    max_z = jnp.nanmax(z_values, axis=1)

    return intersects, min_z, max_z

import jax.numpy as jnp

def transform_points_back(points, theta, phi):
    """
    Transform the array of points back to the original coordinate system.

    Args:
        points: array of shape (..., 3), where ... can be any number of dimensions
        theta: rotation angle around the Z-axis (in radians)
        phi: rotation angle around the Y-axis (in radians)

    Returns:
        Array:
            - points_original: array of same shape as 'points', points in the original coordinate system
    """
    # Compute the rotation matrices
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    Rz = jnp.array([[cos_theta, -sin_theta, 0],
                    [sin_theta,  cos_theta, 0],
                    [0,          0,         1]])

    Ry = jnp.array([[ cos_phi, 0, sin_phi],
                    [ 0,       1, 0      ],
                    [-sin_phi, 0, cos_phi]])

    # Combined rotation matrix
    R = Ry @ Rz

    # Inverse rotation matrix (transpose of R)
    R_inv = R.T

    # Reshape points to (-1, 3) for matrix multiplication
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)  # Shape: (num_points, 3)

    # Apply the inverse rotation
    points_original_flat = points_flat @ R_inv.T  # Shape: (num_points, 3)

    # Reshape back to original shape
    points_original = points_original_flat.reshape(original_shape)

    return points_original

vertical_line_intersects_box_vamp = jax.vmap( vertical_line_intersects_box,in_axes=[None,0,0])

def get_extragalactic_positions(npix:int,theta,phi,x_length:float=20,y_length:float=20,z_length:float=5,num_int_points:int=512,):
    """
    Calculate the integration points along each line of sight for a ``HEALPix`` map with given `nside`, from the location of the observer to the box boundary. A ``HEALPix`` map with a given ``nside`` should contains ``npix = 12*nside**2`` pixels.

    Args:
        npix (int): number of pixels along a axis of the desired rectagular observed map.
        theta : In unit of radian. degrees around Z-axis
        phi : In unit of radian. degrees around Y-axis
        x_length (float): In unit of kpc. half of the box length along x-axis.
        y_length (float): In unit of kpc. half of the box length along y-axis.
        z_length (float): In unit of kpc. half of the box length along z-axis.
        num_int_points (int): the number of integration points along one LoS.

    Returns:
        tuple:
            - poss (jnp.Array): In unit of kpc. 3D array of shape (``npix, num_int_points``, 3), coordinates of integration points along all sightlines of a ``HEALPix`` map.
            - dls (jnp.Array): In unit of kpc. 1D array of shape (``npix``), length of integration segment for all sightlines.
            - nhats (jnp.Array): In unit of rad. 2D array of shape (``npix``,3), unit vector of LoS for all pixels.
            - mask (bool): whether this sightline in the rectagular observed map intersects with the box. If not, leave as 0.
            - resolution (float): the resolution of the rectagular observed map
    """
    x_dim, y_dim, z_dim = 20.0, 20.0, 5.0

    # Rotation angles in radians
    theta_angle = theta  #  degrees around Z-axis
    phi_angle = phi    #  degrees around Y-axis

    # Get rotated vertices
    rotated_vertices = get_rotated_box_vertices(x_dim, y_dim, z_dim, theta_angle, phi_angle)

    # Find the minimum rectangle in X-Y plane
    minx,maxx,miny,maxy = find_min_rectangle_xy(rotated_vertices)
    
    #n_side = 128
    side_length = max(maxx,maxy)
    resolution = side_length*2/npix 

    x = jnp.linspace(-1*side_length,side_length,npix)
    y = x.copy()
    
    x_coords,y_coords = jnp.meshgrid(x,y,indexing='ij')
    
    mask,zmin,zmax = vertical_line_intersects_box_vamp(rotated_vertices,x_coords.reshape(-1),y_coords.reshape(-1))
    zmin = zmin[mask].reshape(-1)
    zmax = zmax[mask].reshape(-1)
    mask = mask.reshape(-1)
    
    x_intersect = x_coords.reshape(-1)[mask]
    y_intersect = y_coords.reshape(-1)[mask]
    z_coords = jnp.linspace(zmax,zmin,num_int_points).T
    
    N = x_intersect.shape[0]
    # Expand x and y to shape (N, Nint)
    x_expanded = x_intersect[:, None]  # Shape: (N, 1)
    y_expanded = y_intersect[:, None]  # Shape: (N, 1)

    # Broadcast x and y to match the shape of z_values
    x_coords = jnp.broadcast_to(x_expanded, (N, num_int_points))  # Shape: (N, Nint)
    y_coords = jnp.broadcast_to(y_expanded, (N, num_int_points))  # Shape: (N, Nint)

    # Stack x_coords, y_coords, and z_values along the last axis
    points = jnp.stack([x_coords, y_coords, z_coords], axis=-1) 
    dls = (zmax-zmin)/num_int_points
    
    nhat = jnp.array([0,0,1.])
    nhats = jnp.broadcast_to(nhat,(N,3))
    
    originral_coords = transform_points_back(points,theta_angle,phi_angle)
    originral_nhats = transform_points_back(nhats,theta_angle,phi_angle)
    
    return originral_coords.transpose((2,0,1)),dls,originral_nhats,mask,resolution