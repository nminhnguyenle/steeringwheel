import cv2
import mediapipe as mp
import math
import time
from pynput.keyboard import Controller, Key
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

keyboard = Controller()
steering_neutral = True
program_active = False

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image = cv2.flip(image, 1)
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    
    
    cv2.line(image, (width//2, 0), (width//2, height), (0, 255, 0), 1)
    cv2.line(image, (0, height//2), (width, height//2), (0, 255, 0), 1)
    
    marker_point = None
    
    if results.multi_hand_landmarks:
      hand_centres = []
      for hand_landmarks in results.multi_hand_landmarks:
        hand_centres.append([int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)])
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      if len(hand_centres) == 2:
        cv2.line(image, (hand_centres[0][0], hand_centres[0][1]), (hand_centres[1][0], hand_centres[1][1]), (0,255,0),5)
        center_x = (hand_centres[0][0] + hand_centres[1][0]) // 2
        center_y = (hand_centres[0][1] + hand_centres[1][1]) // 2
        
        radius = int(math.sqrt((hand_centres[0][0] - hand_centres[1][0]) ** 2 + (hand_centres[0][1] - hand_centres[1][1]) ** 2) / 2)
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 5)
        
        dx = hand_centres[1][0] - hand_centres[0][0]
        dy = hand_centres[1][1] - hand_centres[0][1] 
        slope = int(dy/(dx+1))
        if program_active:
          if abs(slope) > sensitivity:
              if slope < 0:  
                keyboard.press('w')
                keyboard.press('a')
                keyboard.release('d')
              elif slope > 0:
                keyboard.press('w')
                keyboard.press('d')
                keyboard.release('a')
          if abs(slope) < sensitivity:
              keyboard.press('w')
              keyboard.release('a')
              keyboard.release('d')

    sensitivity = 0.3

    
    # Display program status
    status_text = "ACTIVE" if program_active else "INACTIVE"
    status_color = (0, 255, 0) if program_active else (0, 0, 255)
    cv2.putText(image, f"Status: {status_text}", (width//2 - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(image, "Press 'p' to toggle", (width//2 - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('steering simulation', image)
    
    # Handle key presses
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
    elif key == ord('p'): 
        program_active = not program_active
        if not program_active:
            keyboard.release('a')
            keyboard.release('d')

keyboard.release('a')
keyboard.release('d')
keyboard.release('w')
cap.release()
cv2.destroyAllWindows()