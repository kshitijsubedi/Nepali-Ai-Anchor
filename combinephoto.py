import cv2
import numpy as np
sketch = "./proc/"
photo = "./orig/"

for x in range(1,101):
    sketch = sketch + str(x) + ".jpg"
    photo = photo + str(x)+ ".jpg"
    result = "./dataset/" + str(x) + ".jpg"
    img1 = cv2.imread(sketch)
    img2 = cv2.imread(photo)
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(result, vis)
    sketch = "./proc/"
    photo = "./orig/"
