import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import keras

from tensorflow.python.keras import backend as k
from datagenerator_test import DataGenerator


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

#     classes = {
#      'Asparagus, string beans & brussels sprouts': 0,
#      'Bananas, apples & pears': 1,
#      'Bell peppers, zucchinis & eggplants': 2,
#      'Berries & cherries': 3,
#      'Broccoli, cauliflowers, carrots & radish': 4,
#      'Cheese': 5,
#      'Citrus fruits': 6,
#      'Cucumber, tomatoes & avocados': 7,
#      'Eggs': 8,
#      'Fish': 9,
#      'Fresh bread': 10,
#      'Fresh herbs': 11,
#      'Kiwis, grapes & mango': 12,
#      'Lunch & Deli Meats': 13,
#      'Milk': 14,
#      'Minced meat & meatballs': 15,
#      'Nectarines, peaches & apricots': 16,
#      'Onions, leek, garlic & beets': 17,
#      'Pineapples, melons & passion fruit': 18,
#      'Pork, beef & lamb': 19,
#      'Potatoes': 20,
#      'Poultry': 21,
#      'Pre-baked breads': 22,
#      'Pudding, yogurt & quark': 23,
#      'Salad & cress': 24
#       }
    
    classes = dict((v,k) for k,v in classes.items())

    
    df = pd.read_csv('./test.tsv', sep='\t', header=0)
    list_IDs = df['file'].tolist()
    print('Total no. of test data ',(len(df.index)))

    params = {'target_shape': (331,331),
            'batch_size': 1,
            'n_classes': 25,
            'n_channels': 3,
            'shuffle': False,
            'img_dir': './test'}


    # Generators
    test_generator = DataGenerator(list_IDs, **params)

    my_model = keras.models.load_model(args.path, compile=False)
    #test_generator.reset()

    pred_list = []
    filenames = []
    for x, y in test_generator:
        pred = my_model.predict(x)
        print(y)
        print(classes[int(np.squeeze(np.argmax(pred,axis=1)))])
        pred_list.append(classes[int(np.squeeze(np.argmax(pred,axis=1)))])
        filenames.append(y)

    results=pd.DataFrame({"file":filenames,
                      "label":pred_list})

    results.to_csv(args.out,index=False, sep='\t')
    print("Inference done")
  

def parser():
	parser = argparse.ArgumentParser(description='Testing Module')
    
	parser.add_argument('-p', '--path', type=str, default='./model/epoch-024-train_acc-0.9992.h5',
						help='path to saved model')
	parser.add_argument('-o', '--out', type=str, default='./result.tsv',
						help='path to output file')
	return parser


if __name__ == '__main__':
	main(parser().parse_args())