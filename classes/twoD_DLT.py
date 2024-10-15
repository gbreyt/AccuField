"""
Author: Insight in Plain Sight
Date: April 2022

Description: This module provides functions to estimate the homography matrix 
             using the Direct Linear Transform (DLT) method. It computes a 
             homography based on corresponding points from a source image and 
             a target image.

Key Features:
- Find homography matrix from source and target points.
- Construct matrix A used in the DLT algorithm.

Dependencies:
- NumPy

Notes:
- This method is copied from the Medium article: 
  https://medium.com/@insight-in-plain-sight/estimating-the-homography-matrix-with-the-direct-linear-transform-dlt-ec6bbb82ee2b
"""
import numpy as np

def find_homography(points_source, points_target):
    """
    Compute the homography matrix from source points to target points using 
    the Direct Linear Transform (DLT) method.
    
    :param points_source: Corresponding points from the source image.
    :param points_target: Corresponding points from the target image.
    :return: Homography matrix.
    """
    A = construct_A(points_source, points_target)
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # Solution to H is the last column of V, or last row of V transpose
    homography = vh[-1].reshape((3, 3))
    return homography / homography[2, 2]

def construct_A(points_source, points_target):
    """
    Construct matrix A for the homography calculation.
    
    :param points_source: Source points.
    :param points_target: Target points.
    :return: Constructed matrix A.
    """
    assert points_source.shape == points_target.shape, "Shape does not match"
    num_points = points_source.shape[0]

    matrices = []
    for i in range(num_points):
        partial_A = construct_A_partial(points_source[i], points_target[i])
        matrices.append(partial_A)        
    return np.concatenate(matrices, axis=0)

def construct_A_partial(point_source, point_target):
    """
    Construct the partial matrix A for a pair of corresponding points.
    
    :param point_source: A single point from the source.
    :param point_target: A single point from the target.
    :return: Partial matrix A for the given points.
    """
    x, y, z = point_source[0], point_source[1], 1
    x_t, y_t, z_t = point_target[0], point_target[1], 1

    A_partial = np.array([
        [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
        [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
    ])
    return A_partial