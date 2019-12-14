
import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
for k in range(1,1000):
    image= cv2.imread("frames/"+str(k)+".jpg")
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(img)
    for face in faces:

        landmarks = predictor(img, face)
        p1=landmarks.part(49).x
        p2=landmarks.part(49).y
        p3=landmarks.part(55).x
        p4=landmarks.part(55).y
        cv2.rectangle(image,(p1-15,p2-15),(p3+15,p4+15),(0,0,0),-1)
        # for n in range(48, 59):
        #     cv2.line(image,  (landmarks.part(n).x,landmarks.part(n).y), (landmarks.part(n+1).x,landmarks.part(n+1).y), (255, 255, 255), 1)
        # for n in range(60, 67):
        #     cv2.line(image,  (landmarks.part(n).x,landmarks.part(n).y), (landmarks.part(n+1).x,landmarks.part(n+1).y), (255, 255, 255), 1)

        # cv2.line(image,  (landmarks.part(60).x,landmarks.part(60).y), (landmarks.part(67).x,landmarks.part(67).y),(255, 255, 255), 1)
        # cv2.line(image,  (landmarks.part(48).x,landmarks.part(48).y), (landmarks.part(59).x,landmarks.part(59).y),(255, 255, 255), 1)
        # cv2.line(image,  (landmarks.part(48).x,landmarks.part(48).y), (landmarks.part(60).x,landmarks.part(60).y),(255, 255, 255), 1)
        # cv2.line(image,  (landmarks.part(64).x,landmarks.part(64).y), (landmarks.part(54).x,landmarks.part(54).y),(255, 255, 255), 1)
        #image=cv2.resize(image,(256,256))
        cv2.imwrite("blf/"+str(k)+".jpg",image)
        print(k)
