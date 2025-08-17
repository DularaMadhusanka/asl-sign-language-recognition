import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pyttsx3

engine = pyttsx3.init()
engine.say("Test audio output")
engine.runAndWait()
model = load_model('saved_models/20250815_184817/best_model.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

IMG_SIZE = (224, 224)
cap = cv2.VideoCapture(0)  

roi_size = 300
roi_x = 100
roi_y = 100

last_spoken = None
last_spoken_time = 0
speak_cooldown = 1.0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(roi_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                roi, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )
    
        processed_img = cv2.resize(roi, IMG_SIZE)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        processed_img = np.expand_dims(processed_img, axis=0)
        processed_img = preprocess_input(processed_img)  # Use MobileNetV2 preprocessing
        
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        if confidence > 0.1:  
            engine.say("Testing audio in loop")
            engine.runAndWait()
            sign_text = class_names[predicted_class]
            cv2.putText(frame, f"Sign: {sign_text}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Predicted: {sign_text}, Confidence: {confidence}")  # Debug print

            current_time = time.time()
            if (sign_text != last_spoken) or (current_time - last_spoken_time > speak_cooldown):
                engine.say(sign_text)
                engine.runAndWait()
                last_spoken = sign_text
                last_spoken_time = current_time

    cv2.putText(frame, "Place your hand in the green box", (50, frame.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('ASL Recognition', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()