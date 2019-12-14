
import cv2
import numpy as np
import imageio

dim=(460,480)
# ii=cv2.imread("sketch/1.jpg")
# print(ii.shape)

for x in range(1,7450):
    sketch = cv2.imread('blacked/%d.jpg'% x ,cv2.IMREAD_UNCHANGED)
    photo = cv2.imread('frames/%d.jpg'% x ,cv2.IMREAD_UNCHANGED)
    #sketch = cv2.resize(sketch, dim, interpolation = cv2.INTER_AREA)
    #photo = cv2.resize(photo, dim, interpolation = cv2.INTER_AREA)
    result = "dataset/" + str(x) + ".jpg"
    vis = np.concatenate((sketch,photo), axis=1)
    cv2.imwrite(result, vis)
    print(">>",x)

