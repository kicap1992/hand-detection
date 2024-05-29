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

# Landmark names
landmark_names = [str(landmark) for landmark in mp_hands.HandLandmark]


# Loop through camera frames
while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks with MediaPipe
    results = hands.process(frame_rgb)

    # List to store the landmarks of hand 1 and hand 2
    landmarks_hand1 = []
    landmarks_hand2 = []

    # Loop through detected hands
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            middle_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

            # Determine the text position based on hand index and orientation
            if wrist_y < middle_finger_y:
                # Draw landmarks for the detected hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Calculate the bounding box around the hand landmarks
                x_min = int(min(hand_landmarks.landmark, key=lambda lm: lm.x).x * frame.shape[1])
                y_min = int(min(hand_landmarks.landmark, key=lambda lm: lm.y).y * frame.shape[0])
                x_max = int(max(hand_landmarks.landmark, key=lambda lm: lm.x).x * frame.shape[1])
                y_max = int(max(hand_landmarks.landmark, key=lambda lm: lm.y).y * frame.shape[0])

                # Display "Hand 1" or "Hand 2" text on top of the detected hand
                hand_text = f'Hand {idx + 1}'
                # if hand_text == 'Hand 1': hand_name = "Tangan untuk dipijat" else: hand_name = "Tangan yang melakukan pijat"
                if hand_text == 'Hand 1':
                    hand_name = "Tangan untuk dipijat"
                else:
                    hand_name = "Tangan yang melakukan pijat"
                text_x = x_min
                text_y = y_min - 10
                cv2.putText(frame, hand_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # Extract landmarks for each hand
            if idx == 0:
                landmarks_hand1 = hand_landmarks.landmark
                # hand1_text = f'Hand {idx + 1}'
            elif idx == 1:
                landmarks_hand2 = hand_landmarks.landmark
                # hand2_text = f'Hand {idx + 1}'

     

    # Check if landmarks are available for both hands
    if landmarks_hand1 and landmarks_hand2:
        # Get the thumb tip of hand 2
        thumb_tip_hand2 = landmarks_hand2[mp_hands.HandLandmark.THUMB_TIP]

        # Calculate distances between thumb tip of hand 2 and all landmarks of hand 1
        distances = [np.sqrt((landmark.x - thumb_tip_hand2.x)**2 + 
                             (landmark.y - thumb_tip_hand2.y)**2) for landmark in landmarks_hand1]
        
        # print(distances)

        # # Find the index of the closest landmark
        # closest_landmark_idx = np.argmin(distances)

        # # Display the index of the closest landmark
        # cv2.putText(frame, f'Thumb Tip of Hand 2 closest to Landmark {closest_landmark_idx}',
        #             (int(thumb_tip_hand2.x * frame.shape[1]), int(thumb_tip_hand2.y * frame.shape[0])),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Find the index of the closest landmark
        closest_landmark_idx = np.argmin(distances)

        print(np.min(distances))
        
        # Get the name of the closest landmark
        closest_landmark_name = landmark_names[closest_landmark_idx]

        # Display the name of the closest landmark
        cv2.putText(frame, f'Thumb Tip of Hand 2 closest to Landmark: {closest_landmark_name}',
                    (int(thumb_tip_hand2.x * frame.shape[1]), int(thumb_tip_hand2.y * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

   

    # Show frame with detected hands and messages
    cv2.imshow('Hand Detection with Proximity Check', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
