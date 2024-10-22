import numpy as np
import cv2
import mss
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
import time

# Load the pre-trained model
model = tf.keras.models.load_model('facialRecognition_model.keras')

# Screen capture setup
monitor = {"top": 50, "left": 300, "width": 1200, "height": 900}  # Define the screen area to capture

# Preprocess function for each frame
def preprocess_frame(frame, target_size=(224, 224)):
    # Convert the frame (which is in numpy array format) into an image
    img = Image.fromarray(frame)

    # Convert to RGB to remove alpha channel if present
    img = img.convert("RGB")
    
    # Resize the image to the target size (224, 224) as expected by the model
    img = img.resize(target_size)

    # Convert the image to an array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image if needed (depends on your model preprocessing)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    return img_array

import os

# Create a directory to save captured images if it doesn't exist
os.makedirs('captured_images', exist_ok=True)

# Counter for saving images
image_counter = 0

# Real-time emotion prediction
try:
    with mss.mss() as sct:
        while True:
            # Capture the screen
            screen_frame = np.array(sct.grab(monitor))

            # Preprocess the frame
            processed_frame = preprocess_frame(screen_frame)

            # Make a prediction
            predictions = model.predict(processed_frame)

            # Get the predicted emotion (assuming your model gives a label as output)
            predicted_emotion = np.argmax(predictions)

            # Map the predicted emotion index to the corresponding emotion label
            emotion_labels = ["neutral", "happy", "sad", "anger", "surprise", "disgust", "contempt"]
            emotion_text = emotion_labels[predicted_emotion]

            # Create a separate window for displaying the predicted emotion
            emotion_window = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(emotion_window, f'Predicted Emotion: {emotion_text}', 
                        (10, 50),  # Position of the text on the window
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,  # Font scale
                        (255, 255, 255),  # Text color (white)
                        2)  # Thickness of the text

            # Display the captured screen
            cv2.imshow('Screen Capture', screen_frame)
            cv2.imshow('Emotion Display', emotion_window)

            # Save the captured frame every 10 iterations
            if image_counter % 10 == 0:
                cv2.imwrite(f'captured_images/screen_frame_{image_counter}.png', screen_frame)

            image_counter += 1

            # Optional: Add a small delay
            time.sleep(0.1)

            # Exit if 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Process interrupted by user.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cv2.destroyAllWindows()