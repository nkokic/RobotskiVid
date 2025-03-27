import cv2
import numpy as np
import json

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

file = open("camera_params.json")
camera_params = json.load(file)

camera_matrix = np.array(camera_params['camera_matrix'])
dist_coeffs = np.array(camera_params['dist_coeffs'])

#Create the circle
colour = (0,255,255)
lineWidth = -1
radius = 5
point = (0,0)

points = list()
 
#function for detecting left mouse click
def click(event, x,y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed", x,y)
        point = (x,y)
        points.append(point)
        cv2.circle(img_undistorted_cap, point,radius,colour,lineWidth)
        cv2.imshow("Frame", img_undistorted_cap)
         
#event handler
cv2.namedWindow("Frame")      #must match the imshow 1st argument
cv2.setMouseCallback("Frame", click)

img_undistorted_cap = None

while True:
    c = cv2.waitKey(15)

    ret, frame = cap.read()
    img_clone = frame.copy()

    img_undistorted = cv2.undistort(img_clone, camera_matrix, dist_coeffs, None)
    cv2.imshow('Undistorted view', img_undistorted)

    if c == ord('p'):
        img_undistorted_cap = img_undistorted.copy()
        points.clear()
        cv2.imshow("Frame", img_undistorted_cap)

    if c == 27:
        break
