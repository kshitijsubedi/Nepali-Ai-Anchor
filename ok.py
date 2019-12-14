from numpy import load
import numpy as np
import cv2
import dlib 


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


dd=load('data.npz')
ok=dd['arr_0']*2.5
print(ok.shape)


########
# realmidx realmidy nikalne
# predmidx predmidy niklne
# transx = realmidx - predmidx
# transy = realmidy - realmidy
# ani data.npzkoarray[:,:,0] = data.npzkoarray[:,:,0] + transx
# for translation
# ani data.npzkoarray[:,:,1] + data.npzkoarray[:,:,1] + transy
# ########

col=(255,255,255)
th=1

for j in range(1,185):
    image=cv2.imread('frames/%d.jpg'%j)
    #print(image.shape)
    image1=cv2.imread('blf/%d.jpg'%j)

    faces = detector(image)

    for face in faces:
        landmarks = predictor(image, face)
        p1=landmarks.part(48).x
        p2=landmarks.part(48).y
        p3=landmarks.part(54).x
        p4=landmarks.part(54).y

        rmidx=(p3-p1)/2.0
        realmidx = p1 + rmidx
        rmidy=(p4-p2)/2.0
        realmidy = p2+ rmidy

       # print("real",realmidx,realmidy)
    
    zz,xx=ok[j][48]
    cc,vv=ok[j][54]
    pmidx= (cc-zz)/2
    pmidy= (vv-xx)/2
    predmidx = zz + pmidx
    predmidy = xx + pmidy


    #print("predi",predmidx,predmidy)
    #print(ok[j][:,0])

   # cv2.circle(image,(int(predmidx),int(predmidy)),2,(0,0,255),2)
    #cv2.circle(image,(int(realmidx),int(realmidy)),2,(0,0,255),2)
    # cv2.line(image,(int(zz),int(xx)),(int(cc),int(vv)),(255,0,0),1)
   # cv2.line(image,(int(p1),int(p2)),(int(p3),int(p4)),(255,0,0),1)
    transx= realmidx - predmidx
    transy= realmidy - predmidy
    #print(transx,transy)

    ok[j][:,0]= ok[j][:,0] + (transx)
    ok[j][:,1]= ok[j][:,1] + (transy)
    #print(ok[j][:,0])

    #### aba plot 
    zz,xx=ok[j][48]
    cc,vv=ok[j][54]
    bb,nn=ok[j][67]
    aa,ss=ok[j][59]
    dd,ff=ok[j][60]
    gg,hh=ok[j][64]
    ii,oo=ok[j][55]

    for i in range(48, 59):
        z,y=ok[j][i]
        c,v=ok[j][i+1]
        cv2.line(image1, (int(z),int(y)), (int(c),int(v)), color=col, thickness=th)

    cv2.line(image1, (int(zz),int(xx)), (int(aa),int(ss)), color=col, thickness=th)
    cv2.line(image1, (int(zz),int(xx)), (int(dd),int(ff)), color=col, thickness=th)
    cv2.line(image1, (int(cc),int(vv)), (int(gg),int(hh)), color=col, thickness=th)
    cv2.line(image1, (int(bb),int(nn)), (int(dd),int(ff)), color=col, thickness=th)


    for i in range(60, 67):
        z,y=ok[j][i]
        c,v=ok[j][i+1]
        cv2.line(image1, (int(z),int(y)), (int(c),int(v)), color=col, thickness=th)

    ####################

    cv2.imwrite('ffs/%d.jpg'%j,image1)
    print(j)



    
