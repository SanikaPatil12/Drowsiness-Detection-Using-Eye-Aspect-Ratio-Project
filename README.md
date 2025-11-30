# Drowsiness-Detection-Using-Eye-Aspect-Ratio-Project

import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def getLandmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    image.flags.writeable = False
    results = face_mesh.process(image)
    landmarks = []
    if results != None:
        landmarks = results.multi_face_landmarks[0].landmark
    return landmarks, results

def drawFaceMesh(image, results):
    image.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe FaceMesh', image)

def getRightEye(image, landmarks):
    eye_top = int(landmarks[263].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye

def getRightEyeRect(image, landmarks):
    eye_top = int(landmarks[385].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[380].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])

    cloned_image = image.copy()
    cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h

def getLeftEye(image, landmarks):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye

def getLeftEyeRect(image, landmarks):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])

   cloned_image = image.copy()
    cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_left_eye.shape

   x = eye_left
    y = eye_top
    return x, y, w, h

# For IP webcam URL:
ip_camera_url = "http://192.168.4.26:8080/video"
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

success, image = cap.read()
imgb = np.zeros_like(image, dtype="uint8")

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
rightEyeWidth = 0
rightEyeHeight = 0
leftEyeWidth, leftEyeHeight = 0, 0

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()

   if not success:
            print("Ignoring empty camera frame.")
            continue
      imgb = np.zeros_like(image, dtype="uint8")
        landmarks, results = getLandmarks(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 image.flags.writeable = True
        rightEyeImg = getRightEye(image, landmarks)
        rightEyeHeight, rightEyeWidth, _ = rightEyeImg.shape
 xRightEye, yRightEye, rightEyeWidth, rightEyeHeight = getRightEyeRect(image, landmarks)
        cv2.rectangle(image, (xRightEye, yRightEye), (xRightEye + rightEyeWidth, yRightEye + rightEyeHeight),
                      (200, 21, 36), 2)

 # LEFT EYE
  leftEyeImg = getLeftEye(image, landmarks)
        leftEyeHeight, leftEyeWidth, _ = leftEyeImg.shape

  xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight = getLeftEyeRect(image, landmarks)
        cv2.rectangle(image, (xLeftEye, yLeftEye), (xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight), (200, 21, 36),
                      2)

   rightEyeAspectRatio = (rightEyeHeight) / (rightEyeWidth)
        leftEyeAspectRatio = (leftEyeHeight) / (leftEyeWidth)

   eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        print(eyeAspectRatio)

   # Drowsiness detection logic
  if (eyeAspectRatio < 0.2):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 6):
                status = "SLEEPING !!!"
                color = (255, 0, 0)

   elif (eyeAspectRatio >= 0.2 and eyeAspectRatio < 0.3):
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 6):
                status = "Drowsy !"
                color = (0, 0, 255)

   else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 6):
                status = "Active :)"
                color = (0, 255, 0)

   cv2.putText(image, status, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow('MediaPipe FaceMesh', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
explain this code
