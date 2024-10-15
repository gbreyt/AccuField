"""
Authors: Chen and Little
Date: 2019

Description: This module provides utility functions for rotating 3D coordinates 
             around the X, Y, and Z axes. It includes methods for constructing 
             rotation matrices based on specified angles of rotation.

Key Features:
- Rotate coordinates around the X, Y, and Z axes.
- Compute the combined rotation matrix for pan and tilt movements.

Dependencies:
- NumPy
- math
- OpenCV 

Notes:
- Most of the file is directly copied from Chen and Little's work, within their paper "Sports Camera Calibration via Synthetic Data," published in 2019.
- The original file can be accessed here: https://github.com/lood339/SCCvSD/blob/master/python/util/rotation_util.py
- Their GitHub repository: https://github.com/lood339/SCCvSD/tree/master
- Chen, J., Little, J.J.: Sports camera calibration via synthetic data. In: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) Workshops, pp. 2497--2504. IEEE/CVF, Long Beach (CA), USA (2019).
https://doi.org/10.1109/CVPRW.2019.00305

"""

import numpy as np
import math

import cv2 as cv

class RotationUtil:
    @staticmethod
    def rotate_x_axis(angle):
        """
        rotate coordinate with X axis
        https://en.wikipedia.org/wiki/Rotation_matrix + transpose
        http://mathworld.wolfram.com/RotationMatrix.html
        :param angle: in degree
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_y_axis(angle):
        """
        rotate coordinate with X axis
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_z_axis(angle):
        """
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        r = np.transpose(r)
        return r

    @staticmethod
    def pan_y_tilt_x(pan, tilt):
        """
        Rotation matrix of first pan, then tilt
        :param pan:
        :param tilt:
        :return:
        """
        r_tilt = RotationUtil.rotate_x_axis(tilt)
        r_pan = RotationUtil.rotate_y_axis(pan)
        m = r_tilt @ r_pan
        return m
