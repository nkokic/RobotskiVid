import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import convert_2d_points_to_3d_points as conv

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

file = open("LV3\\camera_params.json")
camera_params = json.load(file)
P = np.atleast_2d(camera_params['camera_params'])

imageL = cv.imread("LV3\\imageL.bmp", cv.IMREAD_GRAYSCALE)
imageR = cv.imread("LV3\\imageR.bmp", cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
kpL, desL = sift.detectAndCompute(imageL,None)
kpR, desR = sift.detectAndCompute(imageR,None)

# BFMatcher with default params
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(desL, desR, k=2)
 
# Apply ratio test
good = []
pointsL = []
pointsR = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)
        pointsR.append(kpR[m.trainIdx].pt)
        pointsL.append(kpL[m.queryIdx].pt)

draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=None,
                    flags=2)

image_stereo = cv.drawMatches(imageL, kpL, imageR, kpR, good, None, **draw_params)
plt.imshow(image_stereo)
plt.show()

pointsL = np.int32(pointsL)
pointsR = np.int32(pointsR)
F, mask = cv.findFundamentalMat(pointsL, pointsR, cv.RANSAC)

pointsL = pointsL[mask.ravel()==1]
pointsR = pointsR[mask.ravel()==1]

good_filtered = []
for i in range(len(good)):
    if mask[i]:
        good_filtered.append(good[i])

filtered_kpL = []
filtered_kpR = []
for match in good_filtered:
    filtered_kpL.append(kpL[match.queryIdx])
    filtered_kpR.append(kpR[match.trainIdx])

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
linesL = cv.computeCorrespondEpilines(pointsR.reshape(-1,1,2), 2,F)
linesL = linesL.reshape(-1,3)
img5,img6 = drawlines(imageL,imageR,linesL,pointsL,pointsR)
 
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
linesR = cv.computeCorrespondEpilines(pointsL.reshape(-1,1,2), 1,F)
linesR = linesR.reshape(-1,3)
img3,img4 = drawlines(imageR,imageL,linesR,pointsR,pointsL)
 
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

E = P.T @ F @ P

print(E)

points_3d = conv.convert_2d_points_to_3d_points(filtered_kpL, filtered_kpR, E, P)

with open("LV3\\points_3d.json", "w") as f:
    json.dump(points_3d.tolist(), f)