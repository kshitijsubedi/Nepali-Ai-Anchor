
# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import cv2


model = load_model('models.h5')

for i in range(1,184):
	src_image = cv2.imread('ffs/%d.jpg'% i)
	src_image = cv2.resize(src_image, (256,256), interpolation = cv2.INTER_AREA)
	src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
	src_image = (src_image - 127.5)/127.5
	src_image = expand_dims(src_image,0)
	print(i)
	gen_image = model.predict(src_image)
	gen_image = (gen_image + 1) / 2.0
	pyplot.imshow(gen_image[0])
	pyplot.savefig('gen/%d.jpg'%i)
	#pyplot.axis('off')
	#pyplot.show()
	#gen_image = array_to_img(gen_image[0])

