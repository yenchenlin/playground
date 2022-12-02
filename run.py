import numpy as np
import numpy.linalg as LA


def lookat_to_c2w(lookat, origin, up=[0,1,0]):
    '''
        Calculates the c2w matrix from lookat, origin, and up.
​
        Args:
            origin (arr): [X, Y, Z] world coordinates (in canonical frame) of where camera origin is located.
            lookat (arr): [X, Y, Z] world coordinates (in canonical frame) of where camera should look/point.
            up (arr): [_, _, _] 1 in the location that corresponds to the axis that defines the "up" direction.
​
        Returns:
            c2w (ndarray): 4x4 camera to world matrix
    '''
    row3_arr = [lookat[i] - origin[i] for i in range(len(lookat))]
    # row3 = np.array([[row3_arr[0]],[row3_arr[1]],[row3_arr[2]]])
    row3 = np.array([row3_arr])
    row3 /= np.linalg.norm(row3) # normalized vector 3
    # up = np.array([[up[0]],[up[1]],[up[2]]])
    up = np.array([up])
    row1 = np.cross(up, row3)
    row1 /= np.linalg.norm(row1)
    row2 = np.cross(row3, row1)
    tx = np.array([[origin[0]]])
    ty = np.array([[origin[1]]])
    tz = np.array([[origin[2]]])
    row1 = np.hstack((row1, tx))
    row2 = np.hstack((row2, ty))
    row3 = np.hstack((row3, tz))
    row4 = np.array([[0,0,0,0]])
    c2w = np.vstack((row1, row2, row3, row4))
    return c2w

def get_rays_np(H, W, K, c2w):
    """
    Generate rays of each pixel given camera pose c2w and camera intrinsics K.
    Rays originate at the camera end at a pixel in the HxW image.
​
    Parameters:
        H (int): The height of the image.
        W (int): The width of the image.
        K (ndarray): A 3x3 matrix that stores the intrinsics.
        c2w (ndarray): A 4x4 matrix that stores the camera pose.
​
    Returns:
      (ndarray) Rays' origins in canonical coordinate frame.
      (ndarray) Rays' directions in canonical coordinate frame.
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1) # (u-ox)/fx = X/Z (in camera B frame) [X/Z, -Y/Z, -1]
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def line_sphere_intersection_numpy(totem_pos, totem_radius, rays_d, rays_o, use_min=True):
    '''
        Calculate the camera ray intersection with a totem.
            Equation from here
            https://stackoverflow.com/questions/32571063/specific-location-of-intersection-between-sphere-and-3d-line-segment
​
        Args:
            totem_pos: 3D position of the spherical totem's center
            totem_radius: totem radius in centimeters
            rays_d: (X01, Y01, Z01) in the above link, size (N,3)
            rays_o: (X0, Y0, Z0) in the above link, size (N,3)
            use_min: True if choosing the intersection closer to camera, False if otherwise
        Returns:
            pts: 3D position of the intersections
            valid_idx: indices of the remaining rays (the ones not intersecting get filtered out)
    '''

    # Quadratic formula
    if len(rays_o.shape) == 3:
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(H*W, 3)
        rays_d = rays_d.reshape(H*W, 3)

    shift = rays_o-totem_pos #(N, 3)
    a = np.sum(rays_d**2, axis=1) #(N,)
    b = 2 * np.sum(shift*rays_d, axis=1) #(N,)
    c = np.sum(shift**2, axis=1) - totem_radius**2 #(N,)
    sqrt_inner = b**2 - 4*a*c

    # Filter invalid rays (not intersecting or one intersection)
    valid_idx = np.where(sqrt_inner > 0)[0]

    # Select one of the intersections (ts, distance along rays_d)
    t1 = (-b+np.sqrt(sqrt_inner))/(2*a)
    t2 = (-b-np.sqrt(sqrt_inner))/(2*a)
    ts = [t1, t2]
    if use_min: # first refraction, intersection closer to camera
        t = np.min(ts, axis=0) # (N,)
    else: # second refraction, intersection farther from camera
        t = np.max(ts, axis=0)

    # Compute the 3D position of intersections
    pts = rays_o + t[:, None] * rays_d

    # return_pts = np.zeros((rays_o.shape[0], 3), dtype=object)
    # indices_labeled = np.isin(np.arange(totem_rays_o.shape[0], valid_idx_2))
    # for i, elem in enumerate(indices_labeled):
    #     if not elem:
    #         return_pts[i] = np.array([[None, None]])
    #     else:
    #         return_pts[i] = rays_o[i] + t[:,None] * rays_d[i]

    return pts, valid_idx
    # return return_pts, valid_idx

def line_plane_intersection_numpy(H, W, K, c2w, rays_d, rays_o, use_min=True):
    # Assuming camera image plane lies parallel to XY plane:
    #TODO: generalize, currently hardcoded for the plane Z=4
    # Solution from https://education.siggraph.org/static/HyperGraph/raytrace/rayplane_intersection.htm
    C = 1
    planevec = np.array([[0,0,C]]) # Plane is defined by planevec*(X,Y,Z) + D = 0
    D = -4

    if len(rays_o.shape) == 3:
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(H*W, 3)
        rays_d = rays_d.reshape(H*W, 3)
    #TODO: valid_idx =
    t = ((np.matmul(-planevec,rays_o.T)-D)/(np.matmul(planevec,rays_d.T))).reshape(H*W,1).flatten() # (N,)
    pts = rays_o + t[:,None] * rays_d
    return pts

def get_refracted_ray_numpy(S1, N, n1, n2):
    '''
        Compute the ray direction after refraction.
            Formula from here: http://www.starkeffects.com/snells-law-vector.shtml
        Args:
            S1: incoming ray direction, unit vector
            N: surface normal, unit vector
            n1: refraction index of the medium the ray came from
            n2: refraction index the ray is going into
        Returns:
            the refracted ray direction, array size (N, 3)
    '''
    return n1/n2 * np.cross(N, np.cross(-N, S1)) - N * np.sqrt(1-n1**2/n2**2 * np.sum(np.cross(N, S1) * np.cross(N, S1), axis=1))[:, None]

def cam_rays_to_totem_rays_numpy(cam_rays_o, cam_rays_d, totem_pos, totem_radius, ior_totem, ior_air=1.0):
    '''
        Figure 3 in paper.
        Convert camera rays to totem rays (two refractions).
        Filter out rays that don't intersect with the totem in 3D space.
        Args:
            cam_rays_o: camera ray origins
            cam_rays_d: camera ray directions
            totem_pos: 3D position of the spherical totem's center, numpy
            totem_radius: totem radius in centimeters
            W, H: image width and height
            K: 3x3 camera intrinsic matrix
            ior_totem: totem's index of refraction
            ior_air: air's index of refraction
        Returns:
            totem_rays_o: rays existing from totem origins
            totem_rays_d: rays exiting from totem directions
            valid_idx_1: valid indices after the 1st refraction
            valid_idx_2: valid indices (of valid_idx_1) after the 2nd refraction
            valid_idx_3: valid indices (of valid_idx_2) after the view frustum projection
    '''

    #cam_ray_o = cam_rays_o[0]
    H, W, _ = cam_rays_o.shape
    cam_ray_o = cam_rays_o.reshape(H*W, 3)
    # The first refraction
    D, valid_idx_1 = line_sphere_intersection_numpy(totem_pos, totem_radius, cam_rays_d, cam_rays_o)
    OD = (D-cam_ray_o)/LA.norm(D-cam_ray_o, axis=1)[:, None]
    AD = (D-totem_pos)/LA.norm(D-totem_pos, axis=1)[:, None]
    DE= get_refracted_ray_numpy(OD, AD, ior_air, ior_totem)
    # The second refraction
    E, valid_idx_2 = line_sphere_intersection_numpy(totem_pos, totem_radius, DE, D, use_min=False)
    EA = (totem_pos-E)/LA.norm(totem_pos-E, axis=1)[:, None]
    direction = get_refracted_ray_numpy(DE, EA, ior_totem, ior_air)
    totem_rays_o = E
    totem_rays_d = direction

    # Filter out totem rays that are outside of view frustum
    # valid_idx_3 = get_valid_rays_view_frustum_projection(W, H, K, near, totem_rays_o, totem_rays_d)
    # totem_rays_o = totem_rays_o[valid_idx_3]
    # totem_rays_d = totem_rays_d[valid_idx_3]

    # return_totem_rays_o = np.zeros((H*W, 3), dtype=object)
    # return_totem_rays_d = np.zeros((H*W, 3), dtype=object)

    # indices_labeled = np.isin(np.arange(totem_rays_o.shape[0], valid_idx_2))
    # for i, elem in enumerate(indices_labeled):
    #     if not elem:
    #         return_totem_rays_o[i] = np.array([[None,None]])
    #         return_totem_rays_d[i] = np.array([[None,None]])
    #     else:
    #         return_totem_rays_o[i] = totem_rays_o[i]
    #         return_totem_rays_d[i] = totem_rays_d[i]

    return totem_rays_o, totem_rays_d, valid_idx_1, valid_idx_2#, valid_idx_3

def unravel(pts, H, W):
    '''
            Convert [H*W,3] array (reshaped from [H,W,3] in C-major order) back to [H,W,3]
                pts: (ndarray) of points, size H*W, 3
            Returns:
               pts: (ndarray) of points, size H,W,3
    '''
    return pts.reshape(H,W,3)


if __name__ == "__main__":
    H = 10
    W = 10
    f = 100
    o_x = 0
    o_y = 0
    tx = 0
    ty = -0.8
    tz = -0.7
    totem_center = [0.0, -0.831, 0.8]
    totem_r = 0.2  # mm? cm?
    K = np.array([[f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    c2w = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    # c2w = np.array([
    #     [-1., 0., 0., 0.],
    #     [0., 0.99966649, 0.02582472, -0.831],
    #     [0., 0.02582472, -0.99966649, 0.5],
    #     [0., 0., 0., 0.]
    # ])
    K2 = K
    lookat2 = [0.0,0.0,0.0]
    origin2 = [0.0,0.0,4.0]
    c2w_front = lookat_to_c2w(lookat2, origin2)
    rays_o, rays_d = get_rays_np(H, W, K, c2w) # in canonical frame
    # pts, valid_idx = line_sphere_intersection_numpy(totem_center, totem_r, rays_d, rays_o) # in canonical frame
    totem_rays_o, totem_rays_d, valid_idx_1, valid_idx_2 = cam_rays_to_totem_rays_numpy(rays_o, rays_d, totem_center, totem_r, ior_totem=1.504) # in canonical frame
    pts = line_plane_intersection_numpy(H, W, K2, c2w_front, totem_rays_d, totem_rays_o)
    pts = unravel(pts, H, W)
    print(pts.shape)

    # Merge valid idx
    valid_idx = [idx for idx in valid_idx_1 if idx in valid_idx_2]
    # TODO: use the valid idx to find invalid idx, use the invalid idx to identify bad pts and set them to zero.  