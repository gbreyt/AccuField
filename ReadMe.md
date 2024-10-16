# Soccer Camera Calibration GUI

This GUI is an additional tool created to accompany the paper written by G. Breytenbach and J. Grobler, entitled **"Evaluating the accuracy of a generic field template for camera calibration in soccer broadcast footage,"** submitted to a special issue of *Springer Nature Computer Science*. The issue's name is *Computational and Medical Sciences in Sports*.

## Abstract

Camera calibration is a fundamental process for essential sports analytics tasks, including augmented reality, player tracking, and scene reconstruction. In soccer, camera calibration aims to estimate the geometric relationship between the field and broadcast footage. The international guidelines for soccer fields, however, permit a size variance of up to 1850 m² in soccer fields. This paper investigates whether a generic virtual template can serve as a calibration object for soccer broadcast footage from any internationally approved fields. An experiment is conducted to assess if the Fédération Internationale de Football Association (FIFA) recommended field size can be adapted to fit any internationally-approved field. An initial experiment is conducted with regards to four extreme fields and an arbitrary camera view, after which the experiment is enlarged to cover a thousand camera views for all integer-based allowable field shapes. The direct linear transform is utilised to establish a homography between the generic template and the extreme fields. The initial findings indicate that the generic template can achieve accuracies of at least 93%, as calculated by the standard metric; however, the extended analysis indicated that some arbitrary camera perspectives limit the accuracy to less than 90%. This accuracy metric, however, considers the overall area of the field rather than its distinct segments. Consequently, in addition to the standard metrics employed in the literature, a novel approach is proposed to calculate the combined intersection over union (IoU) as the average IoU per field segment within the visible plane.

## Features

- **Camera Calibration**: Utilise a generic virtual template to calibrate soccer broadcast footage.
- **Field Shape Analysis**: Assess the accuracy of calibration across different soccer field shapes.
- **Homography Calculation**: Employ the Direct Linear Transform (DLT) for homography matrix estimation.


## Installation
1. Clone the repository:
```
git clone https://github.com/gbreyt/AccuField
```

3. Navigate to the folder:
```
cd AccuField
```

5. Ensure Python 3.11 is installed and (optionally) create a new virtual environment with Python@3.11.

6. Install the package requirements:
   Python 3.11.7 was used, however, it has been noted that a Python version newer than 3.8 should suffice.
```
pip install -r requirements.txt
```

7. Difficulties installing the packages:
   Please note that some have experienced issues with installing the specific packages listed. If you too experience such issues, please try
```
opencv-python==4.9
PyQT5==5.15
```

## Usage
Run the following command:
```
python GUI.py
```


