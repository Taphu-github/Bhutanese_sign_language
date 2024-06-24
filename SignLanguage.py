import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Function to release the camera and close OpenCV windows
def release_camera():
    cap.release()
    cv2.destroyAllWindows()

# Function to remove the background from an image and convert it to black and white
def remove_background(image):
    lower_bound = np.array([0, 0, 0], dtype="uint8")
    upper_bound = np.array([50, 50, 50], dtype="uint8")
    mask = cv2.inRange(image, lower_bound, upper_bound)
    image_no_bg = cv2.bitwise_and(image, image, mask=mask)
    image_bw = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image_bw, 1, 255, cv2.THRESH_BINARY)
    return thresholded

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Increase min_detection_confidence for better hand landmark detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

labels_dict = {
    0: "./images/0.jpg", 1: './images/1.jpg', 2: './images/2.jpg', 3: './images/3.jpg', 4: './images/4.jpg', 5: './images/5.jpg', 6: './images/6.jpg',
    7: './images/7.jpg', 8: './images/8.jpg', 9: './images/9.jpg', 10: './images/10.jpg', 11: './images/11.jpg', 12: './images/12.jpg'
}

st.markdown(
    "<div style='background-color: black; padding: 10px; text-align: center;font-family: Roboto, sans-serif;'><h1 style='color: white;'>Sign Language Rata Recognition</h1></div>",
    unsafe_allow_html=True,
)

# Create a black navigation bar with a height of 10%
st.markdown(
    "<div style='background-color: black; height: 10%;'></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='background-color: black; padding: 10px; text-align: center; position: fixed; bottom: 0; height: 20%; width: 46.5%; display: flex; justify-content: center; align-items: center; font-family: Roboto, sans-serif;'><p style='color: white; text-align: center;'>Copyright © Group 8 <br> College of Science and Technology</p></div>",
    unsafe_allow_html=True,
)




# Create two columns for layout
col1, col2 = st.columns([2, 3])

# Create placeholders for the character image and the camera frame inside the columns
character_placeholder = col1.empty()
frame_placeholder = col2.empty()

running = True  # Control variable for the loop

# Define frame size and frame rate
frame_width = 500
frame_height = 500
frame_rate = 30

# Set the camera frame resolution
cap.set(3, frame_width)
cap.set(4, frame_height)

while running:
    ret, frame = cap.read()
    


    if not ret:
        st.error("Error: Unable to retrieve frames from the camera.")
        st.stop()
    frame=cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x - min(x_))
                hand_data.append(landmark.y - min(y_))

            data_aux.extend(hand_data)

    # Ensure data is always of length 84
    while len(data_aux) < 84:
        data_aux.extend([0, 0])  # append zero for undetected hand landmarks

    prediction = model.predict([np.asarray(data_aux)])
    predicted_character_image_path = labels_dict[int(prediction[0])]

    if predicted_character_image_path:
        # Load the predicted character image
        predicted_character_image = cv2.imread(predicted_character_image_path)

        # Resize the character image to match the height of the webcam frame
        character_height = frame.shape[0]
        character_width = int((predicted_character_image.shape[1] / predicted_character_image.shape[0]) * character_height)
        predicted_character_image = cv2.resize(predicted_character_image, (character_width, character_height))

        # Create a white background
        white_background = np.ones((character_height, character_width, 3), dtype=np.uint8) * 255

        # Calculate the position to place the character image in the center of the white background
        x_offset = (white_background.shape[1] - predicted_character_image.shape[1]) // 2

        # Place the character image on the white background
        white_background[:, x_offset:x_offset + character_width] = predicted_character_image

        character_placeholder.image(white_background, channels="BGR", use_column_width=True)

    # Use st.image() for displaying the webcam feed
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)



# Release the camera and close any OpenCV windows
release_camera()
