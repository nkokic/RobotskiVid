import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

file = open("camera_params.json")
camera_params = json.load(file)


#Create the circle
colour = (255,50,50)
lineWidth = -1
radius = 5
point = (0,0)

points = [[]]
 
#function for detecting left mouse click
def click(event, x,y, flags, param):
    global point, pressed, ref_img
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        point = (x,y)
        cv2.circle(ref_img, point,radius,colour,lineWidth)
        points.append([x, y])
        print(points)

        if len(points) >= 2:
            ref_img = ref_img[points[0][1]:points[1][1], points[0][0]:points[1][0]]

        cv2.imshow("Reference", ref_img)

        
         
#event handler
cv2.namedWindow("Reference")      #must match the imshow 1st argument
cv2.setMouseCallback("Reference", click)

ref_img = None

while True:
    c = cv2.waitKey(15)

    ret, frame = cap.read()

    threshold = 50

    cv2.imshow('View', frame)

    if c == ord('p'):
        ref_img = frame.copy()
        points.clear()
        cv2.imshow("Reference", ref_img)

    if c == 27:
        break

    if len(points) >= 2:
        break

gray= cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
 
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray,None)

global target_img
target_img = None

while True:
    c = cv2.waitKey(15)

    ret, frame = cap.read()

    cv2.imshow('View', frame)

    if c == ord('p'):
        target_img = frame.copy()
        cv2.imshow("Target", target_img)
        break

gray2 = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
 
kp2, des2 = sift.detectAndCompute(gray2,None)

# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1,des2,k=2)
 
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)


if len(good) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = ref_img.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dst += (w, 0)  # offset for visualization

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(ref_img, kp1, target_img, kp2, good, None, **draw_params)
    img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("result", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()