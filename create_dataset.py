import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

# Initialize MediaPipe hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

# Check if DATA_DIR exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found.")

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):  # Check if it's a directory
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            if img_path.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                if img is None:
                    print(f"Error loading image: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Process image with MediaPipe hands
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(dir_)

# Save data and labels to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
