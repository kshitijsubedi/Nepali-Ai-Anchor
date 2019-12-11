
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import matplotlib.pyplot as plt
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

n=1
def load_images(path, size=(256,512)):
	global n
	src_list, tar_list = list(), list()
	for filename in listdir(path):
		pixels = load_img(path + filename, target_size=size)
		pixels = img_to_array(pixels)
		src_img, tar_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(src_img)
		tar_list.append(tar_img)
		print(n)
		n+=1
	return [asarray(src_list), asarray(tar_list)]

path = "fin/"
[src_images, tar_images] = load_images(path)
print("Load ", src_images.shape, tar_images.shape)
filename = "face.npz"
savez_compressed(filename, src_images, tar_images)
print("Saved dataset: ", filename) 

##### kehi sample data herna ####
data = load('face.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Loaded: ', src_images.shape, tar_images.shape)
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()