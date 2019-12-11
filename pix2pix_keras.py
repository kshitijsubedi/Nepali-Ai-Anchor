## This Trains the Pix2Pix model with the defined epoches on batches .
## npz file is loaded and used for training on the gan_model
## Model is saved on every epoches.


from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import matplotlib.pyplot as plt
import csv
import pandas

#####
from npz import load_images
from image_utils import graphplot
from image_utils import model_save
from image_utils import summarize_performance
from image_utils import load_real_samples
from image_utils import generate_fake_samples
from image_utils import generate_real_samples
from p2p_model import define_discriminator
from p2p_model import define_encoder_block
from p2p_model import decoder_block
from p2p_model import define_generator
from p2p_model import define_gan

#####

# n_epochs = 23
# n_batch = 10

#####

def train(d_model, g_model, gan_model, dataset, n_epochs=23, n_batch=10):
	n_patch = d_model.output_shape[1]
	trainA, trainB = dataset
	bat_per_epo = int(len(trainA) / n_batch)
	n_steps = bat_per_epo * n_epochs
	for i in range(n_steps):
		ep=i/bat_per_epo
		epoch= int(i/bat_per_epo)
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		print('%d .. >%d, d1[%.3f] d2[%.3f] g[%.3f]' % (epoch+1, i+1, d_loss1, d_loss2, g_loss))
		#tensorboard.on_batch_end(i, named_logs(gan_model, g_loss))
		graphplot(ep,d_loss1,d_loss2,g_loss)
	
		if (i+1) % (bat_per_epo) ==0:
			model_save(epoch+1)
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)



dataset = load_real_samples('face.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

######## training suru 
train(d_model, g_model, gan_model, dataset)


def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	for i in range(len(images)):
		pyplot.subplot(1, 3, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(images[i])
		pyplot.title(titles[i])
	pyplot.show()


### test kasto vayo ta model... ##
## randomly loads a image file and test it on the model and plot it.

[X1, X2] = load_real_samples('face.npz')
print('Loaded', X1.shape, X2.shape)
model = load_model('model_t1.h5')
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]


print(src_image.shape)
gen_image = model.predict(src_image)
print(gen_image.shape)
plot_images(src_image, gen_image, tar_image)

# Test on a single(256,256) image.

import imageio
input_img = imageio.imread('17.jpg')
from skimage import transform,io
input_img = transform.resize(input_img, (256,256), mode='symmetric', preserve_range=True)
model= load_model('model_t1.h5')
final_img=model.predict(input_img)
pyplot.imshow(final_img)
model.summary()

