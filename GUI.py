"""
The Graphical User Interface (GUI)

Author: Gerhardt Breytenbach
Date: August 2024

Description:
This file creates a graphical user interface (GUI) to illustrate the effect of camera parameters and soccer pitch dimensions on calibration accuracy. 
The GUI allows users to adjust various camera settings and pitch dimensions using sliders, and it dynamically updates the projected image and the Intersection over Union (IoU) metrics.

Features:
- Sliders for adjusting pitch dimensions (length and width) and camera parameters (pan, tilt, focal length, x, y, z).
- Displays the IoUSeg and IoUPart accuracy metrics.
- Updates the projected image based on slider values in real-time.

Dependencies:
- NumPy
- OpenCV
- Matplotlib
- PyQt5
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QFrame, QSlider, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt

# Custom classes from provided modules
from classes.PitchGen import SoccerPitch
from classes import CPE, twoD_DLT, iou_util

class SoccerProjectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Soccer Pitch Projection GUI')
        self.setGeometry(100, 100, 1200, 600)  # Adjust window size

        # Central Widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Main layout is horizontal

        # Left side: Create a layout to hold IoU labels and Matplotlib canvas
        plot_layout = QVBoxLayout()  # Vertical layout to stack IoU labels and plot

        # Create a frame for IoU labels to give it a card-like appearance
        iou_frame = QHBoxLayout()
        # IoU label (to be placed inside a card)
        self.iou_label = QLabel("IoUSeg Accuracy: ", self)
        self.iouP_label = QLabel("IoUPart Accuracy: ", self)

        # Create frames for each label to give a card-like look
        iou_label_frame = QFrame(self)
        iou_label_frame.setFrameShape(QFrame.StyledPanel)
        iou_label_frame.setStyleSheet("border: 1px solid lightgray; border-radius: 10px; padding: 10px;")
        label_layout = QVBoxLayout(iou_label_frame)
        label_layout.addWidget(self.iou_label)

        iouP_label_frame = QFrame(self)
        iouP_label_frame.setFrameShape(QFrame.StyledPanel)
        iouP_label_frame.setStyleSheet("border: 1px solid lightgray; border-radius: 10px; padding: 10px;")
        labelP_layout = QVBoxLayout(iouP_label_frame)
        labelP_layout.addWidget(self.iouP_label)

        # Add the label frames to the iou_frame layout for side-by-side placement
        iou_frame.addWidget(iou_label_frame)
        iou_frame.addWidget(iouP_label_frame)

        # Add iou_frame to plot_layout
        plot_layout.addLayout(iou_frame)

        # Matplotlib canvas to display the soccer pitch (720px by 480px)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(720, 480)  # Set fixed size for the plot
        plot_layout.addWidget(self.canvas)

        main_layout.addLayout(plot_layout)  # Add plot layout to the left side of the main layout

        # Right side: Controls (sliders and a button)
        controls_layout = QVBoxLayout()  # For stacking sliders and button
        main_layout.addLayout(controls_layout)

        # Add sliders for pitch dimensions (length, width)
        self.length_slider, self.length_value = self.create_slider(100, 110, 105, 'Length')
        self.width_slider, self.width_value = self.create_slider(64, 75, 68, 'Width')
        
        # Add sliders for camera parameters
        self.pan_slider, self.pan_value = self.create_slider(-35, 35, -30, 'Pan')
        self.tilt_slider, self.tilt_value = self.create_slider(-15, -5, -10, 'Tilt')
        self.focal_length_slider, self.focal_length_value = self.create_slider(1500, 4500, 3018, 'Focal Length')
        self.x_slider, self.x_value = self.create_slider(45, 60, 52, 'X')
        self.y_slider, self.y_value = self.create_slider(-65, -25, -45, 'Y')
        self.z_slider, self.z_value = self.create_slider(10, 25, 16, 'Z')

        # Add a spacer to push sliders to the top
        controls_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Button to trigger recalculation (optional, we can also update on slider movement)
        self.update_button = QPushButton('Update Image', self)
        self.update_button.clicked.connect(self.update_image)
        controls_layout.addWidget(self.update_button)

        # Update the image at startup
        self.update_image()

    def create_slider(self, min_val, max_val, initial, label_text):
        """
        Helper function to create a slider along with a label that displays its value
        """
        layout = QVBoxLayout()
        slider_label = QLabel(f"{label_text}: {initial}", self)  # Label that shows the slider's value
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial)

        # Update the label whenever the slider is moved
        slider.valueChanged.connect(lambda value, lbl=slider_label: lbl.setText(f"{label_text}: {value}"))
        slider.valueChanged.connect(self.update_image)  # Automatically update the image on change
        
        layout.addWidget(slider_label)  # Show the value above the slider
        layout.addWidget(slider)

        self.centralWidget().layout().itemAt(1).addLayout(layout)  # Add the layout to the right side (controls layout)
        return slider, slider_label

    def update_image(self):
        # Fetch current values from sliders
        LENGTH = self.length_slider.value()
        WIDTH = self.width_slider.value()
        pan = self.pan_slider.value()
        tilt = self.tilt_slider.value()
        focal_length = self.focal_length_slider.value()
        x = self.x_slider.value()
        y = self.y_slider.value()
        z = self.z_slider.value()

        # Initialize soccer pitch (FIFA standard pitch)
        FIFA_std = SoccerPitch(FIFA_std=True, offset=True)
        FIFA_points, FIFA_lines = FIFA_std.getPitchPointsAndLines()

        # Initialize custom soccer pitch
        myPitch = SoccerPitch(LENGTH, WIDTH, offset=True)
        pitch_WP, pitch_lines = myPitch.getPitchPointsAndLines()

        # Camera parameters
        cam_pars = [pan, tilt, focal_length, x, y, z]
        cam_pars = CPE.custom_ptz_camera(cc = [x, y, z], fl = focal_length, pan = pan, tilt=tilt)
        
        # Projection matrix
        P0 = CPE.calc_Projection(cam_pars)
        H0 = CPE.get_homography(P0)
        base_im = CPE.generate_edge_image(cam_pars, pitch_WP, pitch_lines)

        # Find homography
        coordinate_point_pairs = [i for i in range(len(FIFA_points))]
        # print(coordinate_point_pairs)
        pitch1_IP = CPE.getImagePoints(coordinate_point_pairs=coordinate_point_pairs, TrueP=P0, test_template_pts=pitch_WP)

        # remove image points outside of screen
        # filtered_true_IP, filtered_FIFA_WP, valid_rows = CPE.remove_invalid_image_points(pitch1_IP, FIFA_points)
        # H = twoD_DLT.find_homography(filtered_FIFA_WP, filtered_true_IP)
        H = twoD_DLT.find_homography(FIFA_points, pitch1_IP)
        M, mask = cv2.findHomography(FIFA_points, pitch1_IP, cv2.RANSAC,5.0)

        pred_img_points = CPE.getImagePoints(coordinate_point_pairs=coordinate_point_pairs, TrueP=H, test_template_pts=FIFA_points)
        # remove image points outside image plane
        # filtered_pred_img_points = pred_img_points[valid_rows]
        # pred_img_points = CPE.getImagePoints(coordinate_point_pairs=coordinate_point_pairs, TrueP=H, test_template_pts=filtered_FIFA_IP)
        iou_seg = iou_util.getSegmentCombinedIoU(pitch1_IP, pred_img_points, return_arr=False)
        # Determine the IoUpart by first warping the predicted and true pitch to birdseye view
        iouwarpPred = iou_util.getWarp(H, 68, 105)
        iouwarpTrue = iou_util.getWarp(H0, WIDTH, LENGTH)

        # Calculate IoUpart
        iou_part = iou_util.calcIoU(iouwarpTrue, iouwarpPred)

        # Update IoU label
        self.iou_label.setText(f"IoUseg = {iou_seg*100:.2f}%")
        self.iouP_label.setText(f"IoUpart = {iou_part*100:.2f}%")

        # Display image in Matplotlib
        self.ax.clear()
        custPlot = iou_util.customPolyPlot(image_points=pred_img_points, polyAdd=0)
        my_im = CPE.generate_edge_image(None, FIFA_points, FIFA_lines, H, background_im=base_im, line_color=(230,0,0))
        # self.ax.imshow(custPlot)
        self.ax.imshow(my_im)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SoccerProjectionGUI()
    gui.show()
    sys.exit(app.exec_())