import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Define the lower and upper bounds of the green color in HSV
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Loop through camera frames
while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to HSV for color detection
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect hand landmarks with MediaPipe
    results = hands.process(frame_rgb)

    # Draw landmarks for the first detected hand with palm facing up
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            if wrist_y < mcp_y:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if green color is detected near hand landmarks
                landmarks_near = False
                landmarks_touched = []

                for landmark in mp_hands.HandLandmark:
                    x = int(hand_landmarks.landmark[landmark].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[landmark].y * frame.shape[0])
                    distance = np.sqrt((x - x_green_center)**2 + (y - y_green_center)**2)
                    if distance < 100:
                        landmarks_near = True
                        landmarks_touched.append(landmark)

                # Display text based on whether the green color is near the landmarks
                if landmarks_near:
                    # print landmarks touched length
                    print(len(landmarks_touched))
                    # text = ', '.join(str(landmark) for landmark in landmarks_touched)
                    # dont show many, just show one
                    text = str(landmarks_touched[0])
                    cv2.putText(frame, f'Green Detected Near: {text}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Create a mask for the green color
    green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

    # Find contours in the green mask
    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the largest contour
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    if largest_contour is not None:
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            x_green_center = int(moments["m10"] / moments["m00"])
            y_green_center = int(moments["m01"] / moments["m00"])
        else:
            x_green_center, y_green_center = -1, -1
    else:
        x_green_center, y_green_center = -1, -1

    # Draw green contours on frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Show frame with landmarks and green color
    cv2.imshow('Hand Landmarks and Green Color Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
