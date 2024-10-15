"""
Pitch Generator

Author: Gerdo Breytenbach
Date: August 2024

Description: This module is designed to create an instantiation of a football field (pitch) according to the regulations stipulated by the IFAB. 
             The field dimensions can be set to FIFA standards or randomized for international pitches. 
             The file includes methods for generating the pitch geometry and plotting it using Matplotlib.
Key Features:
    - Creates a football pitch with specified or randomised dimensions.
    - Generates pitch points and lines for graphical representation.
    - Provides methods to plot the points and lines of the pitch.

Dependencies:
    - NumPy
    - Matplotlib

Notes:
    - All units are in metric.
    - Refer to the IFAB regulations: https://www.theifab.com/laws/latest/the-field-of-play/#field-surface
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class SoccerPitch:
    
    def __init__(self, length = None, width = None, FIFA_std:bool = False, offset: bool = False):
        """
        Initialises a SoccerPitch instance with specified or default dimensions.

        Parameters:
            length (float): Length of the pitch in meters. If None, FIFA's recommended length will be assigned.
            width (float): Width of the pitch in meters. If None, FIFA's recommended width will be assigned.
            FIFA_std (bool): If True, sets the pitch dimensions to FIFA recommended values.
            offset (bool): If True, offsets the pitch to move the origin from the midfield to the bottom left.
        """
        # length = x-axis
        # width = y-axis
        # we need 4 corner points, but for that we need the length and width of the pitch
        if FIFA_std: # Set pitch to FIFA recommended ways
            self.pitch_length = 105
            self.pitch_width = 68
        else:
            if length is not None:
                self.pitch_length = length
            else: # if nothing specified, randomise for international pitch
                self.pitch_length = np.random.randint(100, 110) # 105
            if width is not None:
                self.pitch_width = width 
            else: # if nothing specified, randomise for international pitch
                self.pitch_width = np.random.randint(64, 75) #68
        
        self.goal_width = 7.32
        self.mid_circle_radius = 9.15  # Radius of the center circle in meters
        self.goal_outer_area_length = 16.5  # Length of the goal area in meters
        self.goal_outer_area_width = 16.5 + self.goal_width/2 # Width of the goal area in meters
        self.goal_inner_area_length = 5.5
        self.goal_inner_area_width = 5.5 + self.goal_width/2
        self.goal_width = 7.32
        self.penalty_length = 11
        self.penalty_width = 0
        self.mid_x = 0
        self.mid_y = 0
        self.halfmoon_radius = 9.15  # Radius of the penalty box's half-moon in meters

        self.genPitch(offset=offset)

    def genPitch(self, offset:bool = False):
        """
        Generates the pitch geometry, including points and lines that define the pitch layout.

        Parameters:
            offset (bool): If True, applies offset to the generated pitch points.
        """
        pitch_points = np.zeros((245,2), dtype=np.float32)
        pitch_lines = np.zeros((235,2), dtype=np.uint8)

        #* Origin point
        pitch_points[0] = (0, 0)

        #* we need 4 corner points
        # lines from 1 to 2, 1 to 4 AND from 2 to 3, 3 to 4
        pitch_points[1] = (-self.pitch_length/2, -self.pitch_width/2) 
        pitch_points[2] = (-self.pitch_length/2,  self.pitch_width/2)
        pitch_points[3] = ( self.pitch_length/2,  self.pitch_width/2)
        pitch_points[4] = ( self.pitch_length/2, -self.pitch_width/2)

        pitch_lines[0] = (1,2)
        pitch_lines[1] = (1,4)
        pitch_lines[2] = (2,3)
        pitch_lines[3] = (3,4)

        #* we need the middle points for the half-way line
        # line from 5 to 6
        pitch_points[5] = (0,  self.pitch_width/2)
        pitch_points[6] = (0, -self.pitch_width/2)

        pitch_lines[4] = (5,6)

        #* we need the big box points
        # lines from 7 to 8, 8 to 9, 9 to 10
        pitch_points[7] = (-self.pitch_length/2, self.goal_outer_area_width)
        pitch_points[8] = (-self.pitch_length/2 + self.goal_outer_area_length, self.goal_outer_area_width)
        pitch_points[9] = (-self.pitch_length/2 + self.goal_outer_area_length, -self.goal_outer_area_width)
        pitch_points[10] = (-self.pitch_length/2, -self.goal_outer_area_width)

        pitch_lines[5] = (7,8)
        pitch_lines[6] = (8,9)
        pitch_lines[7] = (9,10)
        
        # lines from 11 to 12, 12 to 13, 13 to 14
        pitch_points[11] = (self.pitch_length/2, self.goal_outer_area_width)
        pitch_points[12] = (self.pitch_length/2 - self.goal_outer_area_length, self.goal_outer_area_width)
        pitch_points[13] = (self.pitch_length/2 - self.goal_outer_area_length, -self.goal_outer_area_width)
        pitch_points[14] = (self.pitch_length/2, -self.goal_outer_area_width)

        pitch_lines[8] = (11,12)
        pitch_lines[9] = (12,13)
        pitch_lines[10] = (13,14)

        #* we need the small box points
        # lines from 15 to 16, 16 to 17, 17 to 18
        pitch_points[15] = (-self.pitch_length/2,  self.goal_inner_area_width)
        pitch_points[16] = (-self.pitch_length/2 + self.goal_inner_area_length,  self.goal_inner_area_width)
        pitch_points[17] = (-self.pitch_length/2 + self.goal_inner_area_length, -self.goal_inner_area_width)
        pitch_points[18] = (-self.pitch_length/2, -self.goal_inner_area_width)

        pitch_lines[11] = (15,16)
        pitch_lines[12] = (16,17)
        pitch_lines[13] = (17,18)
        
        # lines from 19 to 20, 20 to 21, 21 to 22
        pitch_points[19] = (self.pitch_length/2,  self.goal_inner_area_width)
        pitch_points[20] = (self.pitch_length/2 - self.goal_inner_area_length,  self.goal_inner_area_width)
        pitch_points[21] = (self.pitch_length/2 - self.goal_inner_area_length, -self.goal_inner_area_width)
        pitch_points[22] = (self.pitch_length/2, -self.goal_inner_area_width)

        pitch_lines[14] = (19,20)
        pitch_lines[15] = (20,21)
        pitch_lines[16] = (21,22)

        #* we need the penalty placement points
        pitch_points[23] = (-self.pitch_length/2 + self.penalty_length, 0)
        pitch_points[24] = ( self.pitch_length/2 - self.penalty_length, 0)

        #* we need the middle circle points, let's plot 120 points in the 360 degree radius
        num_points = 120
        angle_increment = 360 / num_points

        for i in range(num_points):
            angle_degrees = i * angle_increment
            angle_radians = math.radians(angle_degrees)
            x = self.mid_circle_radius * math.cos(angle_radians)
            y = self.mid_circle_radius * math.sin(angle_radians)
            pitch_points[25 + i] = (x, y)
            if i > 0: # at least one point of the circle has to have been plotted
                pitch_lines[16 + i] = (25+i-1, 25+i)


        #* LEFT we need to plot the half-moons on the outer boxes
        num_points = 50
        start_angle = 360 - 53.05
        end_angle = 360 + 54.8
        angle_increment = (end_angle - start_angle) / num_points

        for i in range(num_points):
            angle_degrees = start_angle + i * angle_increment
            angle_radians = math.radians(angle_degrees)
            x = self.halfmoon_radius * math.cos(angle_radians) - self.pitch_length/2 + self.penalty_length
            y = self.halfmoon_radius * math.sin(angle_radians)
            pitch_points[145 + i] = (x, y)
            if i > 0: # at least one point of the circle has to have been plotted
                pitch_lines[135 + i] = (145+i-1, 145+i)
        
        #* RIGHT we need to plot the half-moons on the outer boxes
        num_points = 50
        start_angle = 180 - 53.05
        end_angle = 180 + 55
        angle_increment = (end_angle - start_angle) / num_points

        for i in range(num_points):
            angle_degrees = start_angle + i * angle_increment
            angle_radians = math.radians(angle_degrees)
            x = self.halfmoon_radius * math.cos(angle_radians) + self.pitch_length/2 - self.penalty_length
            y = self.halfmoon_radius * math.sin(angle_radians)
            pitch_points[195 + i] = (x, y)
            if i > 0 and i < 52: # at least one point of the circle has to have been plotted
                pitch_lines[185 + i] = (195+i-1, 195 + i)

        self.pitch_points = pitch_points
        self.pitch_lines = pitch_lines

        if offset:
            self.offSetPoints()

    def getPitchPointsAndLines(self):
        """
        Retrieves the pitch points and lines for further processing or plotting.

        Returns:
            tuple: A tuple containing two elements:
                - pitch_points (numpy.ndarray): An array of points defining the pitch geometry.
                - pitch_lines (numpy.ndarray): An array of lines connecting the pitch points.
        """
        return self.pitch_points, self.pitch_lines
    
    def offSetPoints(self):
        """
        Adjusts the pitch points to offset the origin from midfield to the bottom left of the pitch.
        """
        for i in range(self.pitch_points.shape[0]):
            self.pitch_points[i][0] += self.pitch_length / 2
            self.pitch_points[i][1] += self.pitch_width / 2
        
    def plot_points(self):
        """
        Plots the soccer pitch points using Matplotlib based on the generated pitch points and lines.
        """
        points, lines = self.getPitchPointsAndLines()

        # Extract x and y coordinates from the points
        x_coords, y_coords = zip(*points)

        # Create a scatter plot of the points with numbering
        plt.scatter(x_coords, y_coords, color='blue', label='Points')

        # Numbering each point on the plot
        for i, point in enumerate(points[:25]):
            plt.text(point[0], point[1], f'{i+1}', fontsize=12, ha='center', va='bottom')

        # Set plot labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Scatter Plot of Points with Numbers')

        # Display the plot
        plt.legend()
        plt.grid(True)
        
        plt.show()

    def plot_lines(self, color = "red"):
        """
        Plots the soccer pitch lines using Matplotlib based on the generated pitch points and lines.
        """
        # Get the set of points
        points, lines = self.getPitchPointsAndLines()
        
        for line in lines:
            start_point = points[line[0]]
            end_point = points[line[1]]
            plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)

        # Set plot labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Scatter Plot of Points with Numbers')

        # Display the plot
        plt.legend()
        plt.grid(True)
        
        plt.show()


# Create an instance of SoccerPitch
if __name__ == '__main__':
    soccer_pitch = SoccerPitch(FIFA_std=True, offset=True)
    pts, lines = soccer_pitch.getPitchPointsAndLines()
    soccer_pitch.plot_points()
    soccer_pitch.plot_lines("red")
