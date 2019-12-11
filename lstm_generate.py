#Usage : python generate.py -i /audio-file-path/ -m /model-path/ -d 1 -c 3 -o /output-folder-path/


import tensorflow as tf
import librosa
import numpy as np
from numpy import asarray
from numpy import savez_compressed
import os, shutil, subprocess
from keras import backend as K
from keras.models import Model, Sequential, load_model
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="input speech file")
parser.add_argument("-m", "--model", type=str, help="DNN model to use")
parser.add_argument("-d", "--delay", type=int, help="Delay in terms of number of frames, where each frame is 40 ms")
parser.add_argument("-c", "--ctx", type=int, help="context window size")
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()


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

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx



output_path = args.out_fold
num_features_Y = 136
num_frames = 75
wsize = 0.04
hsize = wsize
fs = 44100
trainDelay = args.delay
ctxWin = args.ctx

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

model = load_model(args.model)


test_file = args.in_file


# Used for padding zeros to first and second temporal differences
zeroVecD = np.zeros((1, 64), dtype=np.longdouble)
zeroVecDD = np.zeros((2, 64), dtype=np.longdouble)

# Load speech and extract features
sound, sr = librosa.load(test_file, sr=fs)
melFrames = np.transpose(melSpectra(sound, sr, wsize, hsize))
melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)

features = np.concatenate((melDelta, melDDelta), axis=1)
features = addContext(features, ctxWin)
features = np.reshape(features, (1, features.shape[0], features.shape[1]))

upper_limit = features.shape[1]
lower = 0
generated = np.zeros((0, num_features_Y))

# Generates face landmarks one-by-one
# This part can be modified to predict the whole sequence at one, but may introduce discontinuities
for i in tqdm(range(upper_limit)):
    cur_features = np.zeros((1, num_frames, features.shape[2]))
    if i+1 > 75:
        lower = i+1-75
    cur_features[:,-i-1:,:] = features[:,lower:i+1,:]

    pred = model.predict(cur_features)
    generated = np.append(generated, np.reshape(pred[0,-1,:], (1, num_features_Y)), axis=0)

# Shift the array to remove the delay
generated = generated[trainDelay:, :]
tmp = generated[-1:, :]
for _ in range(trainDelay):
    generated = np.append(generated, tmp, axis=0)

if len(generated.shape) < 3:
    generated = np.reshape(generated, (generated.shape[0], int(generated.shape[1]/2), 2))
keypoints = generated*100


print(keypoints.shape)



data = asarray(keypoints)

savez_compressed(output_path +'/data.npz', data)