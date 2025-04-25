import cv2
import numpy as np
import json

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

file = open("camera_params.json")
camera_params = json.load(file)

camera_matrix = np.atleast_2d(camera_params['camera_matrix'])
dist_coeffs = np.atleast_2d(camera_params['dist_coeffs'])

#Create the circle
colour = (255,50,50)
lineWidth = -1
radius = 5
point = (0,0)

points = [[]]
 
#function for detecting left mouse click
def click(event, x,y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        cv2.circle(canny_capture, point,radius,colour,lineWidth)
        points.append([x, y])

        if (len(points) >= 6):
            canny_capture[0:points[4][1], :] = 0
            canny_capture[:, 0:points[4][0]] = 0
            canny_capture[points[5][1]:, :] = 0
            canny_capture[:, points[5][0]:] = 0
            
        cv2.imshow("Frame", canny_capture)
         
#event handler
cv2.namedWindow("Frame")      #must match the imshow 1st argument
cv2.setMouseCallback("Frame", click)

img_undistorted_cap = None

while True:
    c = cv2.waitKey(15)

    ret, frame = cap.read()
    img_clone = frame.copy()

    img_undistorted = cv2.undistort(img_clone, camera_matrix, dist_coeffs, None)
    threshold = 50
    canny_img = cv2.Canny(img_undistorted.copy(), threshold, threshold*3)
    cv2.imshow('Undistorted view', canny_img)

    if c == ord('p'):        
        canny_capture = canny_img.copy()
        points.clear()
        cv2.imshow("Frame", canny_capture)

    if c == 27:
        break

    if len(points) >= 6:
        break

# 250 x 170
worldPoints = np.array([[0, 0, 0], [0, 0.17, 0], [0.25, 0, 0], [0.25, 0.17, 0]])

worldPoints = np.asarray(worldPoints, dtype=np.float32).reshape(-1, 3)
imagePoints = np.asarray(points[0:4], dtype=np.float32).reshape(-1, 2)

isSuccess, rotation, position = cv2.solvePnP(worldPoints, imagePoints, camera_matrix, dist_coeffs)

rotation, _ = cv2.Rodrigues(rotation)

A = camera_matrix @ rotation
b = camera_matrix @ position

theta = 0
ro = 1

lambdaX = A[0,0] * np.cos(theta) + A[1,0] * np.sin(theta) - ro * A[2,0]
lambdaY = A[0,1] * np.cos(theta) + A[1,1] * np.sin(theta) - ro * A[2,2]
lambdaRo = b[2] * ro - b[0] * np.cos(theta) - b[1] * np.sin(theta)

newTheta = np.atan2(lambdaY, lambdaX)
newRo = lambdaRo / np.sqrt(lambdaX**2 + lambdaY**2)

print(f"theta: {newTheta}")
print(f"ro: {newRo}")