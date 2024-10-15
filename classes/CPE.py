"""
The Camera Projection Engine

Author: Gerhardt Breytenbach
Date: August 2024

Description:
This file defines the Camera Projection Engine (CPE), which is responsible for projecting coordinates, generating edge images, obtaining projection (or homography) matrices, and converting 3D world coordinates into 2D image coordinates. The code allows for synthetic image generation based on soccer field models and camera parameters, and is a core component of the projection system of the GUI.

Key Features:
- Functions to calculate projection and homography matrices.
- Projections of 3D world coordinates into 2D image coordinates.
- Generation of synthetic edge images based on field templates and camera parameters.
- Incorporation of customized pan, tilt, and zoom (PTZ) camera setups.

Dependencies:
- NumPy
- OpenCV

Note:
Much of this code is based on the methodology introduced by Chen and Little (2019) in their paper, "Sports Camera Calibration via Synthetic Data."
"""


import numpy as np
import cv2
from classes.rotation_util import RotationUtil


def get_homography(P):
    """
    homography matrix from the projection matrix
    :return:
    """
    h = P[:, [0, 1,3]]
    return h

def calc_Projection(camera_params, homography:bool = False):
    # Extract the camera parameters
    u, v, fl = camera_params[0:3]
    rod_rot = camera_params[3:6]
    cc = camera_params[6:9]
    
    H = np.zeros((3, 4)) # Homography
    
    # Calibration (Intrinsic)
    K = np.asarray(
        [[fl, 0, u],
        [0, fl, v],
        [0, 0, 1]]
    )
    
    # Camera center
    camera_center = np.asarray(cc)
    # Rotation?
    rotation = np.asarray(rod_rot)
    # Projection matrix
    P = np.zeros((3, 4))
    for i in range(3):
        P[i][i] = 1.0
        P[i][3] = -camera_center[i]
        
    r, _ = cv2.Rodrigues(rotation)

    P = K @ r @ P
    
    if homography:
        return get_homography(P)
    
    return P

# This function is also adapted from Chen 2019. The adaptation allows for the 3D projection with either a homography or projection matrix
def project_3D(P, x, y, z=0, w=1.0, homography:bool = False):
        # Determine if the matrix is a homography or projection matrix,
        if P.shape[1] == 3:
            homography = True
        # Remove or use the z value accordingly
        if homography:
        # Make a vector 'p' that represents the actual coordinate we want to project
            p = np.zeros(3)
            p[0], p[1], p[2] = x, y, w
        else:
        # Make a vector 'p' that represents the actual coordinate we want to project
            p = np.zeros(4)
            p[0], p[1], p[2], p[3] = x, y, z, w
        
        q = P @ p
        assert q[2] != 0.0
        return (q[0]/q[2], q[1]/q[2])

# This function is designed to create a synthetic edge image
def generate_edge_image(camera_params, model_points, model_line_segment, P = None, im_h=720, im_w=1280, line_width=4, my_homography:bool = False, target_size= None, background_im = None, line_color = (255, 255, 255), dot_template: bool = False):
    # Globals
    # Retrieve the projection matrix
    if P is None:
        P = calc_Projection(camera_params)
        if my_homography == True:
            P = get_homography(P)
        
    # Generate an empty image (black) that will be filled with the lines of the actual field template
    if background_im is None:
        im = np.ones((im_h, im_w, 3), dtype=np.uint8)*230
        if line_color == (255, 255, 255): line_color = (0, 0, 0)
    else:
        im = background_im

    n = model_line_segment.shape[0]
    color = line_color
    
    #Plot all the points and lines on the black image with Projection matrix
    for i in range(n):
        idx1, idx2 = model_line_segment[i][0], model_line_segment[i][1]
        p1, p2 = model_points[idx1], model_points[idx2]
        p1z, p2z = 0.0, 0.0
        if model_points[idx1].size > 2:
            p1z, p2z = model_points[idx1][2], model_points[idx2][2]
        q1 = project_3D(P, p1[0], p1[1], p1z, 1.0)
        q2 = project_3D(P, p2[0], p2[1], p2z, 1.0)
        q1 = np.rint(q1).astype(np.int64)
        q2 = np.rint(q2).astype(np.int64)
        coordinate_shape = type(q1)
        if model_line_segment[i].size > 2: # this means we are dashing the line
            if model_line_segment[i][2] == 1:
                cv2.line(im, tuple(q1), tuple(q2), color, thickness=line_width, lineType=cv2.LINE_AA, shift=0)
            else:
               cv2.line(im, tuple(q1), tuple(q2), color, thickness=line_width) 
        else:
            if dot_template:
                cv2.line(im, tuple(q1), tuple(q2), color, thickness=line_width, lineType=cv2.LINE_AA)
            else:
                cv2.line(im, tuple(q1), tuple(q2), color, thickness=line_width)

    if target_size is not None:
        if target_size[1] is None:
            aspect_ratio = 1280/720
            target_size[1] = int(target_size[0] / aspect_ratio)
            target_size = tuple(target_size)
            
        im = cv2.resize(im, target_size)

    return im


# This funciton returns the image points from a set of world coordinates
def getImagePoints(coordinate_point_pairs, TrueP, test_template_pts, generic_template_pts = None):
    num_coords = len(coordinate_point_pairs)
    img_points = np.empty((num_coords ,2))
    world_points = np.empty((num_coords, 3))    

    for i in range(num_coords):
        img_points[i] = project_3D(TrueP, test_template_pts[i][0], test_template_pts[i][1])
        img_points[i] = np.rint(img_points[i]).astype(np.int64) # Make the image points integeres, probably for easier plotting
        rand_float = (np.random.random()*7) # This is to ensure matrix A in the DLT has a rank of 11, otherwise no DLT convergence (can't be on the same plane)
        if generic_template_pts is not None: world_points[i] = (generic_template_pts[coordinate_point_pairs[i]][0], generic_template_pts[coordinate_point_pairs[i]][1], rand_float)

    if generic_template_pts is not None:
        return img_points, world_points
    else: 
        return img_points
    
def custom_ptz_camera(cc = [52, -45, 16], fl = 3018, pan = -35, tilt = -10):
    """
        This function creates the set of camera parameters to create a camera matrix:
    """
    rolls = 0
    u = 1280/2.0
    v = 720/2

    camera = np.zeros((9))

    base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_x_axis(rolls) @\
            RotationUtil.rotate_x_axis(-90)
    pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pan, tilt)
    rotation = pan_tilt_rotation @ base_rotation
    
    rot_vec, _ = cv2.Rodrigues(rotation)

    camera[0], camera[1] = u, v
    camera[2] = fl
    camera[3], camera[4], camera[5] = rot_vec[0], rot_vec[1], rot_vec[2]
    camera[6], camera[7], camera[8] = cc[0], cc[1], cc[2]

    return camera