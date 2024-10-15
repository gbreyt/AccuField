"""
IoU Analysis

Author: Gerhardt Breytenbach
Date: August 2024

Description:
This file defines functions for evaluating the accuracies of homography matrices in the context of sports broadcast footage camera calibration. It includes capabilities to calculate Intersection over Union (IoU) scores, generate masks for critical areas of the soccer field, and visualise these areas for analysis. The code supports the calibration and assessment of camera projection systems based on synthetic image data.

Key Features:
- Calculation of IoU scores between ground truth and predicted masks.
- Generation of masks for essential field areas, including pitch and penalty areas.
- Customisable polygon plotting for visual analysis of IoU.
- Comprehensive analysis of segment-wise IoU for different field regions.

Dependencies:
- NumPy
- OpenCV

Note:
Some functions in this code are adapted from the code implemented by Chen and Little (2019) in their paper, "Sports Camera Calibration via Synthetic Data."
"""

import cv2
import numpy as np


def getWarp(h, temp_h, temp_w):
    """
    Warp the homography to a birdseye perspective using the given homography matrix.
    
    Parameters:
    - h: Homography matrix to warp the image.
    - temp_h, temp_w: True template height and width of the warp mask.

    Returns:
    - A resized and warped mask of size (105, 68) after applying perspective warp and resizing of true field dimensions.
    """
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    h = np.linalg.inv(h)
    warp_mask = cv2.warpPerspective(img, h, (temp_w, temp_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    if temp_w != 105 or temp_h != 68:
        # Resize the image using interpolation (optional)
        warp_mask = cv2.resize(warp_mask, (105, 68), interpolation=cv2.INTER_CUBIC)  # Choose interpolation method
    return warp_mask


def calcIoU(gt_mask, pred_mask):
    """
    Calculate the Intersection over Union (IoU) between the ground truth and predicted masks.
    
    Parameters:
    - gt_mask: Ground truth mask.
    - pred_mask: Predicted mask.
    
    Returns:
    - IoU score (between 0 and 1).
    """

    val_intersection = (gt_mask != 0) * (pred_mask != 0) 
    val_union = (gt_mask != 0) + (pred_mask != 0)
    u = float(np.sum(val_union))

    if u <= 0:
        iou = 0
    else:
        iou = 1.0 * np.sum(val_intersection) / u
    return iou

def getCriticalAreaPoints(image_points):
    """
    Extract key field area points from the given image points.
    
    Parameters:
    - image_points: Coordinates of the image points.
    
    Returns:
    - Various field areas as numpy arrays: pitch area, penalty area, goal area, penalty arc, etc.
    """

    # Right hand side of the field
    pitch_area = np.array([image_points[i].tolist() for i in [2, 3, 4, 1]], dtype=np.int32)
    penalty_area = np.array([image_points[i].tolist() for i in [12, 13, 14, 11]], dtype=np.int32)
    goal_area = np.array([image_points[i].tolist() for i in [20, 21, 22, 19]], dtype=np.int32)
    penalty_arc = np.array([image_points[i+195].tolist() for i in range(50)], dtype=np.int32)

    # Left hand side of the field
    penalty_area_left = np.array([image_points[i].tolist() for i in [8, 9, 10, 7]], dtype=np.int32)
    goal_area_left = np.array([image_points[i].tolist() for i in [16, 17, 18, 15]], dtype=np.int32)
    penalty_arc_left = np.array([image_points[i+145].tolist() for i in range(50)], dtype=np.int32)

    # Mid circle
    mid_circle = np.array([image_points[i+25].tolist() for i in range(120)], dtype=np.int32)

    return pitch_area, penalty_area, goal_area, penalty_arc, mid_circle, penalty_area_left, goal_area_left, penalty_arc_left

def customPolyPlot(image_points, polyAdd:int = 1, polyRemove: int = None, im_height: int = 720, im_width: int = 1280, colour = (255, 255, 255)):
    """
    Create a customized polygon plot for IoU analysis, optionally adding or removing polygons.

    Parameters:
    - image_points: Points defining the areas.
    - polyAdd: Index of the polygon to add (1-based index).
    - polyRemove: Index of the polygon to remove (1-based index).
    - im_height, im_width: Image dimensions.
    - colour: Colour to fill the added polygon.

    Returns:
    - An image with the specified polygons filled.
    """

    areas = getCriticalAreaPoints(image_points=image_points)
    img = np.zeros((im_height, im_width, 3), np.uint8)
    
    distinct_colors = [
        (230, 0, 0),      # Red
        (0, 230, 0),      # Green
        (0, 0, 230),      # Blue
        (230, 230, 0),    # Yellow
        (230, 0, 230),    # Magenta
        (0, 230, 230),    # Cyan
        (128, 0, 128),    # Purple
        (230, 165, 0)     # Orange
    ]
    
    if polyAdd == 0:
        for i in range(len(areas)):
            cv2.fillPoly(img, [areas[i]], distinct_colors[i])
    else: 
        cv2.fillPoly(img, [areas[polyAdd - 1]], colour)

    if polyRemove is not None: cv2.fillPoly(img, [areas[polyRemove - 1]], (0, 0, 0))

    return img

def getIoUmask(image_points, predicted_points):
    """
    Generate an IoU mask by plotting true and predicted points and converting them to grayscale.

    Parameters:
    - image_points: Ground truth points.
    - predicted_points: Predicted points.

    Returns:
    - IoU score for the given masks.
    """
    mask1_true = customPolyPlot(image_points, 1)
    mask1_pred = customPolyPlot(predicted_points, 1)
    mask1_true = cv2.cvtColor(mask1_true, cv2.COLOR_BGRA2GRAY) 
    mask1_pred = cv2.cvtColor(mask1_pred, cv2.COLOR_BGRA2GRAY) 
    # Add IoU
    return calcIoU(mask1_true, mask1_pred)


def getSegmentCombinedIoU(image_points, predicted_points, return_arr: bool = False):
    """
    Calculate the combined IoU for different segments of the field by masking and comparing them.

    Parameters:
    - image_points: Ground truth points.
    - predicted_points: Predicted points.
    - return_arr: Whether to return an array of individual IoUs.

    Returns:
    - Average IoU across visible segments, or an array of individual IoUs if return_arr is True.
    """

    IoU = 0.0
    cIoU = 0.0
    IoUs = []

    # Calculate IoU for each field segment by selectively masking/removing areas

    #* Step 1 Greater field
    mask1_true = customPolyPlot(image_points, 1, 2)
    mask1_pred = customPolyPlot(predicted_points, 1, 2)
    # Add IoU
    IoU = calcIoU(mask1_true, mask1_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 2 Big box right
    mask2_true = customPolyPlot(image_points, 2, 3) 
    mask2_pred = customPolyPlot(predicted_points, 2, 3)
    # Add IoU
    IoU = calcIoU(mask2_true, mask2_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 3 Goal area right
    mask3_true = customPolyPlot(image_points, 3) # There are no field segments inside this segment, therefore no need to remove something
    mask3_pred = customPolyPlot(predicted_points, 3)
    # Add IoU
    IoU = calcIoU(mask3_true, mask3_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 4 Penalty arc right
    mask4_true = customPolyPlot(image_points, 4) 
    mask4_pred = customPolyPlot(predicted_points, 4)
    # Add IoU
    IoU = calcIoU(mask4_true, mask4_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 5 Middle Circle
    mask5_true = customPolyPlot(image_points, 5) 
    mask5_pred = customPolyPlot(predicted_points, 5)
    # Add IoU
    IoU = calcIoU(mask5_true, mask5_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 6 Big box left
    mask6_true = customPolyPlot(image_points, 6, 7) 
    mask6_pred = customPolyPlot(predicted_points, 6, 7)
    # Add IoU
    IoU = calcIoU(mask6_true, mask6_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 7 Goal area left 
    mask7_true = customPolyPlot(image_points, 7) 
    mask7_pred = customPolyPlot(predicted_points, 7)
    # Add IoU
    IoU = calcIoU(mask7_true, mask7_pred)
    cIoU += IoU
    IoUs.append(IoU)

    #* Step 2 Penalty arc left
    mask8_true = customPolyPlot(image_points, 8) 
    mask8_pred = customPolyPlot(predicted_points, 8)
    # Add IoU
    IoU = calcIoU(mask8_true, mask8_pred)
    cIoU += IoU
    IoUs.append(IoU)

    areas = [
        "pitch right",
        "penalty right",
        "goal right",
        "arc right",
        "mid_circle",
        "penalty left",
        "goal left",
        "arc left"
    ]

    # get the number of visible field segments:
    visible_field_segments = 0
    for i, iou in enumerate(IoUs):
        # print(f"{areas[i]} \t {iou}")
        if iou != 0: # assuming not in image plane
            visible_field_segments += 1

    if visible_field_segments == 0:
        return -0
            

    if return_arr:
        return cIoU/visible_field_segments, IoUs
    return cIoU/visible_field_segments
