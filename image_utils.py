def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
	trainA, trainB = dataset
	ix = randint(0, trainA.shape[0], n_samples)
	X1, X2 = trainA[ix], trainB[ix]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

def generate_fake_samples(g_model, samples, patch_shape):
	X = g_model.predict(samples)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# plot sample images using the ongoing training model

def summarize_performance(step, g_model, dataset, n_samples=3):
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def model_save(ok):
  g_model.save('model-epoch%d.h5' % (ok))


# from keras.callbacks import TensorBoard

#import plotly



# tensorboard = TensorBoard(
#   log_dir='/log/my_tf_logs',
#   histogram_freq=0,
#   batch_size=batch_size,
#   write_graph=True,
#   write_grads=True
# )
# tensorboard.set_model(gan_model)

# def named_logs(model, logs):
#   result = {}
#   for l in zip(model.metrics_names,itertools.repeat(logs)):
#     result[l[0]] = l[1]
#   return result



#saving the losses training in the csv file for plotting later.

def graphplot(epoch,dl1,dl2,gl):
  with open('loss.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([epoch, dl1,dl2,gl])