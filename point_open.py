import cv2
import mediapipe as mp
import numpy as np
import json

# Load JSON file
with open('dataset.json', 'r') as f:
    all_data = json.load(f)

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Landmark names
landmark_names = [str(landmark) for landmark in mp_hands.HandLandmark]

def is_pointing_gesture(landmarks):
    # Index finger tip and other joints
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]


    # Check if the index finger is extended
    index_extended = (index_tip.y < index_dip.y < index_pip.y < index_mcp.y)

    # Check if other fingers are bent
    other_fingers_bent = True
    for finger_tip, finger_pip in [
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)]:
        if landmarks[finger_tip].y < landmarks[finger_pip].y:
            other_fingers_bent = False


    if other_fingers_bent is False:
        return True

    return False



# Loop through camera frames
while True:
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks with MediaPipe
    results = hands.process(frame_rgb)

    pointing_hand = []
    open_palm = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
                
            # Check for gestures
            gesture = None
            if is_pointing_gesture(landmarks):
                gesture = "Pointing"
                # check the idx of the hand
                pointing_hand = hand_landmarks.landmark
            else:
                gesture = "Open Palm"
                open_palm = hand_landmarks.landmark

            # Debugging output
            # print(f"Detected Gesture: {gesture}")

            if gesture:
                # Get the coordinates for the wrist landmark to place the text above the hand
                wrist = landmarks[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                # Add text annotation to the frame
                cv2.putText(frame, gesture, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if gesture == "Pointing":
                    if open_palm:
                        finger_tip = pointing_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        distances = [np.sqrt((landmark.x - finger_tip.x)**2 +
                                            (landmark.y - finger_tip.y)**2) for landmark in open_palm]

                        if distances:
                            closest_landmark_idx = np.argmin(distances)
                            # print("ini index", closest_landmark_idx)
                            
                            dataset_image_path = all_data[closest_landmark_idx]["image_path"]
                            # show image
                            dataset_image = cv2.imread(dataset_image_path)
                            cv2.imshow('Dataset Image', dataset_image)
                            
                            closest_landmark_name = landmark_names[closest_landmark_idx]
                            cv2.putText(frame, f'Closest Landmark: {closest_landmark_name}',
                                        (int(finger_tip.x * w), int(finger_tip.y * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, 'No open palm detected', (cx, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
