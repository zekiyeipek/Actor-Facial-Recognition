# Actor-Facial-Recognition
This project captures screen images in real-time, preprocesses the images, and uses a pre-trained deep learning model to predict facial emotions.

## Features
- Real-Time Screen Capture: Continuously captures a specified portion of the screen using the mss library.
- Emotion Recognition: Utilizes a pre-trained TensorFlow model for recognizing facial emotions from the screen captures.
- Prediction Display: Shows the captured screen and the predicted emotion in real-time via separate OpenCV windows.
- Image Logging: Periodically saves screen captures as images for further analysis or debugging.
- Customizable Capture Area: The screen capture area can easily be modified by adjusting the monitor settings in the script. 

## Pre-requisites
Ensure you have Python 3.x installed on your system. You will also need to have the following Python libraries installed:

- TensorFlow/Keras
- OpenCV
- MSS (Multi-Screen Shot)
- NumPy
- Pillow (PIL)
