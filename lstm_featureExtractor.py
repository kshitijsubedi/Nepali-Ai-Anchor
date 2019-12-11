# Usage: python lstm_featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file

import numpy as np
import cv2
# Use : pip install opencv-python==3.4.2.16
import math
import copy
import dlib
from keras import backend as K
from numpy import asarray
from numpy import savez_compressed
from copy import deepcopy
import sys
import os
import librosa
import imageio
import argparse, fnmatch, shutil
from tqdm import tqdm
import subprocess


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-vp", "--video-path", type=str, help="video folder path")
parser.add_argument("-sp", "--sp-path", type=str, help="shape_predictor_68_face_landmarks.dat path")
parser.add_argument("-o", "--output-path", type=str, help="Output file path")
args = parser.parse_args()

predictor_path = args.sp_path
video_folder_path = args.video_path
dataset_path = args.output_path

print(predictor_path)
print(video_folder_path)
print(dataset_path)


class faceNormalizer(object):
    w = 600
    h = 600

    def __init__(self, w = 600, h = 600):
        self.w = w
        self.h = h

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)
      
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        
        return tform

    def tformFlmarks(self, flmark, tform):
        transformed = np.reshape(np.array(flmark), (68, 1, 2))           
        transformed = cv2.transform(transformed, tform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def alignEyePoints(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = alignedSeq[0,:,:]
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
        eyecornerSrc  = [ (firstFlmark[36, 0], firstFlmark[36, 1]), (firstFlmark[45, 0], firstFlmark[45, 1]) ]

        tform = self.similarityTransform(eyecornerSrc, eyecornerDst);

        for i, lmark in enumerate(alignedSeq):
            alignedSeq[i] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def transferExpression(self, lmarkSeq, meanShape):
        exptransSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = exptransSeq[0,:,:]
        indexes = np.array([60, 64, 62, 67])

        tformMS = cv2.estimateRigidTransform(firstFlmark[:,:], np.float32(meanShape[:,:]), True)

        sx = np.sign(tformMS[0,0])*np.sqrt(tformMS[0,0]**2 + tformMS[0,1]**2)
        sy = np.sign(tformMS[1,0])*np.sqrt(tformMS[1,0]**2 + tformMS[1,1]**2) 
        prevLmark = copy.deepcopy(firstFlmark)
        prevExpTransFlmark = copy.deepcopy(meanShape)

        zeroVecD = np.zeros((1, 68, 2))
        diff = np.cumsum(np.insert(np.diff(exptransSeq, n=1, axis=0), 0, zeroVecD, axis=0), axis=0)
        msSeq = np.tile(np.reshape(meanShape, (1, 68, 2)), [lmarkSeq.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        exptransSeq = diff + msSeq

        return exptransSeq

    def unitNorm(self, flmarkSeq):
        normSeq = copy.deepcopy(flmarkSeq)
        normSeq[:, : , 0] /= self.w
        normSeq[:, : , 1] /= self.h
        return normSeq


def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst

    melspec = np.log(1e-16+librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec
ms = np.load('mean_shape.npy') # Mean face shape
fnorm = faceNormalizer()
ms = fnorm.alignEyePoints(np.reshape(ms, (1, 68, 2)))[0,:,:]

try:
    os.remove(dataset_path)
except:
    print ('Exception when deleting previous dataset...')

wsize = 0.04
hsize = 0.04

zeroVecD = np.zeros((1, 64))
zeroVecDD = np.zeros((2, 64))

speechData = np.empty((0, 75 , 128))
lmarkData = np.empty((0, 75 ,136 ))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

points_old = np.zeros((68, 2), dtype=np.float32)



for root, dirnames, filenames in os.walk(video_folder_path):
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mp4':
            f = os.path.join(root, filename)
            print(f)
            vid = imageio.get_reader(f,  'ffmpeg')
            point_seq = []
            img_seq = []
            for frm_cnt in tqdm(range(0, vid.count_frames())):
                points = np.zeros((68, 2), dtype=np.float32)

                try:
                    img = vid.get_data(frm_cnt)
                except:
                    print('FRAME EXCEPTION!!')
                    continue

                dets = detector(img, 1)
                if len(dets) != 1:
                    print('FACE DETECTION FAILED!!')
                    continue

                for k, d in enumerate(dets):
                    shape = predictor(img, d)

                    for i in range(68):
                        points[i, 0] = shape.part(i).x
                        points[i, 1] = shape.part(i).y

                point_seq.append(deepcopy(points))

            cmd = 'ffmpeg -y -i '+os.path.join(root, filename)+' -vn -acodec pcm_s16le -ac 1 -ar 44100 temp.wav'
            subprocess.call(cmd, shell=True) 

            y, sr = librosa.load('temp.wav', sr=44100)

            os.remove('temp.wav')
            frames = np.array(point_seq)
            fnorm = faceNormalizer()
            aligned_frames = fnorm.alignEyePoints(frames)
            transferredFrames = fnorm.transferExpression(aligned_frames, ms)
            frames = fnorm.unitNorm(transferredFrames)
            if frames.shape[0] != 75:
                continue
            lmarkData = np.append(lmarkData ,np.reshape(frames, (1, 75, 136)) , axis=0)
            print(lmarkData.shape)

            melFrames = np.transpose(melSpectra(y, sr, wsize, hsize))
            melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
            melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
            melFeatures = np.concatenate((melDelta, melDDelta), axis=1)

            if melFeatures.shape[0] != 75:
                continue
            speechData = np.append(speechData ,np.reshape(melFeatures, (1, 75, 128)) , axis=0)
            print(speechData.shape)
            


savez_compressed(dataset_path +'/lmarkData.npz', lmarkData)
savez_compressed(dataset_path +'/speechData.npz', speechData)