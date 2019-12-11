#python train.py -i path-to-train-file/ -u number-of-hidden-units -d number-of-delay-frames -c number-of-context-frames -o output-folder-to-save-model-file
# Training LSTM part of the project

import tensorflow as tf
import librosa
import numpy as np
import os, shutil, subprocess
from keras import backend as K
from keras.layers import Input, LSTM, Dense, Reshape, Activation, Dropout, Flatten
from keras.models import Model
from tqdm import tqdm
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
import argparse, fnmatch
import random
import time, datetime
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="Input file containing train data")
parser.add_argument("-u", "--hid-unit", type=int, help="hidden units")
parser.add_argument("-d", "--delay", type=int, help="Delay in terms of number of frames")
parser.add_argument("-c", "--ctx", type=int, help="context window size")
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()

output_path = args.out_fold+'_'+str(args.hid_unit)+'/'

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

ctxWin = args.ctx
num_features_X = 128 * (ctxWin+1)# input feature size
num_features_Y = 136 # output feature size --> (68, 2)
num_frames = 75 # time-steps
batchsize = 128
h_dim = args.hid_unit
lr = 1e-3


drpRate = 0.2 # Dropout rate 
recDrpRate = 0.2 # Recurrent Dropout rate 

frameDelay = args.delay # Time delay

#load
lmarkData = np.load(args.in_file + "/lmarkData.npz")
speechData = np.load(args.in_file + "/speechData.npz")



flmark = lmarkData['arr_0']
MelFeatures = speechData['arr_0']

numIt = int(flmark.shape[0]//batchsize) + 1
metrics = ['MSE', 'accuracy']

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx


def dataGenerator():
    X_batch = np.zeros((batchsize, num_frames, num_features_X))
    Y_batch = np.zeros((batchsize, num_frames, num_features_Y))

    idxList = list(range(flmark.shape[0]))

    batch_cnt = 0    
    while True:
        random.shuffle(idxList)
        for i in idxList:
            cur_lmark = flmark[i, :, :]
            cur_mel = MelFeatures[i, :, :]

            if frameDelay > 0:
                filler = np.tile(cur_lmark[0:1, :], [frameDelay, 1])
                cur_lmark = np.insert(cur_lmark, 0, filler, axis=0)[:num_frames]
             
            X_batch[batch_cnt, :, :] = addContext(cur_mel, ctxWin)
            Y_batch[batch_cnt, :, :] = cur_lmark
            
            batch_cnt+=1

            if batch_cnt == batchsize:
                batch_cnt = 0
                yield X_batch, Y_batch

def build_model():
    net_in = Input(shape=(num_frames, num_features_X))
    h = LSTM(h_dim, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(net_in)
    h = LSTM(num_features_Y, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    model = Model(inputs=net_in, outputs=h)
    model.summary()

    opt = Adam(lr=lr)

    model.compile(opt, metrics[0], 
                metrics= metrics[1:])
    return model


gen = dataGenerator()
model = build_model()

X, Y = next(gen)

history = model.fit(X, Y, validation_split=0.33, epochs=200 , batch_size=128, verbose=0)
model.save(output_path+'Model.h5')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()