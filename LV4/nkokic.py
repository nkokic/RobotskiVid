import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import read_kinect_pic as lv4

def distance(point, plane):
    return np.abs(point[2] - (point[0] * plane[0] + point[1] * plane[1] + plane[2]))

directory = "LV4\\images\\sl-"
imageName = "00411"
depth_image_name = directory + imageName + "-D.txt"
image = cv.imread(directory + imageName + ".bmp")

print(image.shape)

depth_image, point_3d_array, n_3d_points = lv4.read_kinect_pic(depth_image_name, (image.shape))

point_3d_array = np.atleast_2d(point_3d_array)

x = point_3d_array[:,0].flatten()
y = point_3d_array[:,1].flatten()
z = point_3d_array[:,2].flatten()

error_tolerance = 1

T = []
R = []
for i in range(100):
    print(f"Iteration: {i+1}/100")
    index_sample = np.random.choice(n_3d_points, 3)
    points_sample = point_3d_array[index_sample, :]

    a = points_sample.copy()
    a[:,2] = 1

    b = np.atleast_2d(points_sample[:,2]).T

    r = (np.linalg.inv(a) @ b).flatten()

    t = []
    for p in point_3d_array:
        e = distance(p, r)
        if(e < error_tolerance):
            t.append(p)

    if(len(t) > len(T)):
        R = r
        T = t

print("Done!")

T = np.atleast_2d(T)
print(T)
print(R)

x1 = T[:,0].flatten()
y1 = T[:,1].flatten()
z1 = T[:,2].flatten()

fig = plt.figure()

# ax = fig.add_subplot(1, 1, 1, projection='3d')

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(depth_image)
plt.show()

plt.scatter(x, -y, c=z)
plt.scatter(x1, -y1, c='r')
plt.show()

# fig = plt.figure()

# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(x, z, -y, c=z)
# ax.scatter(x1, z1, -y1, c='r')

# plt.show()