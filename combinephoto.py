## Used to combine the sketches image and the original images into a single image
## The final image is used for the training.


import cv2
import numpy as np

dim=(256,256)
sketch = cv2.imread("/misc/sketch/1.jpg")
cv2.imshow("",sketch)

# for x in range(1,7450):
#     sketch = cv2.imread('/misc/sketch/%d.jpg'% x ,cv2.IMREAD_UNCHANGED)
#     photo = cv2.imread('/misc/frames/%d.jpg'% x ,cv2.IMREAD_UNCHANGED)

#     print(sketch.shape)
#     sketch = cv2.resize(sketch, dim, interpolation = cv2.INTER_AREA)
#     photo = cv2.resize(photo, dim, interpolation = cv2.INTER_AREA)
#     result = "/misc/dataset/" + str(x) + ".jpg"
#     img1 = cv2.imread(sketch)
#     img2 = cv2.imread(photo)
#     vis = np.concatenate((img1, img2), axis=1)
#     cv2.imwrite(result, vis)
#     print(">>")

