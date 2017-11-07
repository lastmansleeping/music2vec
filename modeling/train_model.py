# Import Libraries
import tensorflow as tf
import numpy
import glob
import sys
from genre_classifier import GenreClassifier

_TRAIN = 0.80
_VAL = 0.05
_TEST = 0.15


def split_data(input_path):
	data = {'train': dict(), 'val': dict(), 'test': dict()}
	for genre_path in glob.glob(input_path + '/*'):
		genre = genre_path.split('/')[-1]
		files = glob.glob(genre_path + '/*.npy')
		data['train'][genre] = files[:int(_TRAIN * len(files))]
		data['val'][genre] = files[int(_TRAIN * len(files)):-int(_TEST * len(files))]
		data['test'][genre] = files[-int(_TEST * len(files)):]
	return data


def genre_classifier(data):
	config = tf.ConfigProto(allow_soft_placement=True) # , log_device_placement=True
	config.gpu_options.allow_growth = True
	# config.gpu_options.per_process_gpu_memory_fraction = 0.90
	with tf.Session(config=config) as sess:
		# with tf.device('/device:GPU:0'):
		model = GenreClassifier()
		model.train(sess, data['train'], data['val'])
		accuracy = model.evaluate(sess, data['test'])
		print('***** test accuracy: %.3f' % accuracy)
		saver = tf.train.Saver()
		model_path = saver.save(
            sess, "./models/genre_classifier.ckpt")
		print("Model saved in %s" % model_path)


def main():
	input_path = sys.argv[1]
	data = split_data(input_path)
	genre_classifier(data)


if __name__ == '__main__':
    main()
