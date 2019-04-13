import os
import numpy as np
import pandas as pd
import keras
import argparse
from keras.models import Sequential
from keras.applications.nasnet import NASNetLarge

from datagenerator import DataGenerator

def main(args):

	classes = {
	'Bananas, apples & pears':0,
	'Pre-baked breads':1,
	'Bell peppers, zucchinis & eggplants':2,
	'Cucumber, tomatoes & avocados':3,
	'Pork, beef & lamb':4,
	'Kiwis, grapes & mango':5,
	'Minced meat & meatballs':6,
	'Citrus fruits':7,
	'Cheese':8,
	'Berries & cherries':9,
	'Eggs':10,
	'Broccoli, cauliflowers, carrots & radish':11,
	'Pudding, yogurt & quark':12,
	'Poultry':13,
	'Potatoes':14,
	'Fresh herbs':15,
	'Salad & cress':16,
	'Nectarines, peaches & apricots':17,
	'Lunch & Deli Meats':18,
	'Pineapples, melons & passion fruit':19,
	'Milk':20,
	'Fresh bread':21,
	'Asparagus, string beans & brussels sprouts':22,
	'Onions, leek, garlic & beets':23,
	'Fish':24
	}


	# Read data and preprocess
	df = pd.read_csv('./train.tsv', sep='\t', header=0)
	df['label'].value_counts()
	df['class'] = df['label'].map(classes)
	#df.head()
	list_IDs = df['file'].tolist()
	labels = dict(zip(df['file'], df['class']))
	print('Total no. of training data ',(len(df.index)))

	# Parameters
	params = {'target_shape': (331,331),
			'batch_size': args.batch_size,
			'n_classes': 25,
			'n_channels': 3,
			'shuffle': True,
			'img_dir': './train'}


	# Generators
	training_generator = DataGenerator(list_IDs, labels, **params)

   # Design model
	nasnetmodel = NASNetLarge(include_top=False, weights='imagenet')

	model = Sequential()
	model.add(nasnetmodel)
	model.add(keras.layers.GlobalAveragePooling2D(name='avg_pool'))
	model.add(keras.layers.Dense(25, activation='softmax', name='fc'))


	# model compile
	opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# callbacks
	filepath = './'+args.outdir+'/epoch-{epoch:03d}-train_acc-{acc:.4f}.h5'
	save_model = keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max', period=1)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

	# Train model on dataset
	model.fit_generator(generator=training_generator, epochs = args.epoch, verbose = 1, steps_per_epoch=500, callbacks=[save_model,reduce_lr])
	model.save('./{args.outdir}/my_model.h5')


def parser():
	parser = argparse.ArgumentParser(description='Training Module')
   
	parser.add_argument('-i', '--epoch', type=int, default=50,
						help='No. of epochs ')
	parser.add_argument('-b', '--batch-size', type=int, default=8,
						help='batch size ')
	parser.add_argument('-o', '--outdir', type=str, default='./model',
						help='directory where models are saved')

	return parser


if __name__ == '__main__':
	main(parser().parse_args())