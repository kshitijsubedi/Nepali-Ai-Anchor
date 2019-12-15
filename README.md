<<<<<<< HEAD
"# Nepali Ai Anchor" "#Nepali-Ai-Anchor"
=======
"# Nepali Ai Anchor" 
"# Nepali-Ai-Anchor" 
>>>>>>> ee252fc19cef9210eea810e050d383c2228d5a02
# First Nepali AI Anchor 


Given Nepali Unicode text of a news article, we synthesize a high quality video of an anchor presenting the content provided in the input text with professional news broadcasting backdrop. Trained on many hours of a person narrating news articles, a recurrent neural network learns the mapping from audio generated from the input text to mouth shapes which then is used to synthesize high quality mouth texture, and composite it to what he might have looked pronouncing the input texts.


>This project was submitted to Itonics Hackathon 2019. 

# The project depends on the following Python packages:
- Keras --- 2.2.5
- Tensorflow --- 1.15.0
- Librosa --- 0.6.0
- opencv-python --- 3.4.2.16
- dlib --- 19.7.0
- tqdm
- subprocess

It also depends on the following packages:
- ffmpeg --- 3.4.1

The code has been tested on Windows 10 and Google colab.


# Feature Extraction For LSTM
You can run lstm_featureExtractor file to extract features from videos directly. The arguments are as follows:

The dataset used to train lstm is GridCorpus.
In the cmd below $i is the speaker number
```sh

$ "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip"
$ unzip -q "video/s$i.zip" -d "../video"
```
- -vp --- Input folder containing video files (if your video file types are different from .mpg - or .mp4, please modify the script accordingly)
- -sp --- Path to shape_predictor_68_face_landmarks.dat. You can download this file [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat).
- -o --- Output file name

Usage:
```sh
$ python featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file-folders
```

# Train LSTM
The training code has the following arguments:

- -i --- Input file containing folder with training data
- -u --- Number of hidden units
- -d --- Delay in terms of frames, where one frame is 40 ms
- -c --- Number of context frames
- -o --- Output folder path to save the model
- 
Usage:
```sh
$ python lstm_train.py -i path-to-train-file/ -u number-of-hidden-units -d number-of-delay-frames -c number-of-context-frames -o output-folder-to-save-model-file
```

# LSTM Generate
The generation code has the following arguments:

- -i --- Input speech file
- -m --- Input talking face landmarks model
- -d --- Delay in terms of frames, where one frame is 40 ms
- -c --- Number of context frames
- -o --- Output path


Usage:
```sh
$ python lstm_generate.py -i /audio-file-path/ -m /model-path/ -d 1 -c 3 -o /output-folder-path/ 
```

<<<<<<< HEAD





=======
>>>>>>> ee252fc19cef9210eea810e050d383c2228d5a02
