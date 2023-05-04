# Driver Drowsiness Detection using Deep Learning-based Model

This GitHub repository contains an implementation of a driver drowsiness detection system using deep learning-based models. The system is based on two models: the first one takes patches for the right and left eyes and classifies them as closed or opened, while the second one is responsible for classifying if the person is yawning or not through classifying mouth patches.

## Models

### Eye Closed/Open Detection Model

The eye closed/open detection model is based on transfer learning with the VGG16 model. We added a new classification layer, froze the rest of the pre-trained layers, and used the Kaggle "yawn_eye_dataset_new" dataset for training. The model takes patches for the right and left eye as input and classifies them as closed or opened.

### Yawning Detection Model

The yawning detection model is developed in a similar way to the eye closed/open detection model. The model takes patches for the mouth as input and classifies them as yawning or not.

## Inference

For inference, we use frames captured from a live stream. We use `dlib.get_frontal_face_detector()` to get faces in the frame and then use `dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')` to get the 68 points forming the face landmark so we can determine patches for the right eye, left eye, and mouth to be fed to the model. The same pre-processing applied through training process is applied to the frames before inference.

## Post-Processing

For post-processing, we count the number of frames for which the first model predicted closed and the number of frames the second model predicted yawning. If they exceed a certain threshold, a drowsiness alert is produced.

## Getting Started

### Prerequisites

To run the code in this repository, you will need:

- Python 3 installed on your local machine
- Required Python packages installed (numpy, pandas, matplotlib, tensorflow, keras, opencv, dlib, imutils, pygame)

You can install the required packages using the following command:

`pip install numpy pandas matplotlib tensorflow keras opencv-python dlib imutils pygame`

You will also need to download the "shape_predictor_68_face_landmarks.dat" file and place it in the root directory of the repository. This file is used by the dlib library to detect facial landmarks in the input frames. You can download the file from the following link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Note that the `mixer` package from `pygame` is used to play a sound alert when drowsiness is detected. If you do not wish to use this feature, you can comment out the relevant code in the `drowsiness_detection.py` file.

### Running the Code

To run the code, navigate to the root directory of the repository and run the following command:

`python drowsiness_detection.py`

This will start the driver drowsiness detection system. You can adjust the threshold values for the number of frames with closed eyes and yawning by modifying the `score` and `score_mouth` variables in the code.

