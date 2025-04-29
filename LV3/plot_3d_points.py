import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cv2

def main():
    # Load and prepare image
    imageL = cv2.imread("LV3/imageL.bmp")
    imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
    h, w, _ = imageL.shape

    # Load intrinsic matrix P (3x3)
    file = open("LV3\\camera_params.json")
    camera_params = json.load(file)
    P = np.array(camera_params['camera_params'])

    # Create a grid of pixel coordinates
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    pixels_hom = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=0)  # shape (3, N)

    # Depth at which to place the image plane
    z = -7.0

    # Backproject pixel rays into 3D space (scale by z)
    P_inv = np.linalg.inv(P)
    points_3d = (P_inv @ pixels_hom) * z  # shape (3, N)

    # Reshape for plotting
    X = points_3d[0].reshape(h, w)
    Y = points_3d[1].reshape(h, w)
    Z = points_3d[2].reshape(h, w)


    fig = plt.figure(figsize=plt.figaspect(0.5))

    points_3d_path = 'LV3/points_3d.json'

    with open(points_3d_path, 'r') as f:
        points_3d = np.array(json.load(f))


    x = points_3d[:,0].flatten()
    y = points_3d[:,1].flatten()
    z = points_3d[:,2].flatten()

    tri = mtri.Triangulation(x, y)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(-x, -y, -z, triangles=tri.triangles, cmap=plt.cm.Spectral)
    ax.plot_surface(X, Y, Z, facecolors=imageL / 255.0, shade=False)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(-x, -y, -z)
    ax.plot_surface(X, Y, Z, facecolors=imageL / 255.0, shade=False)

    plt.show()

if __name__ == '__main__':
    main()