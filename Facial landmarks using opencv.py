import os
import numpy as np
import cv2
import face_recognition as fg
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw
import dlib

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for f in faces:
        x1 = f.left()
        y1 = f.top()
        x2 = f.right()
        y2 = f.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        landmarks = predictor(gray, f)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (0, 215, 255), -1)

    cv2.imshow("window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
