""" In this project, I will identify dog breeds by their pictures.  To do that I'll be using Keras + TensorFlow.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

import os
import sys
import keras
from keras.preprocessing import image
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

DIR = "keras/dog_breed_identification/"
TEST_DIR = "keras/dog_breed_identification/test/"
TRAIN_DIR = "keras/dog_breed_identification/train/"
EPOCHS = 75

def announce():
	print ('*******************************************')
	print ('** Launching Dog Breed Identification... **')
	print ('*******************************************')
	

def preprocess_train_data (train_df):

	# Prepare images. Use size 32X32x3.
    X_train = np.zeros((train_df.shape[0], 32, 32, 3))
    count=0
    
    for fig in train_df['id']:
        img = image.load_img(TRAIN_DIR+fig+".jpg", target_size=(32, 32, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    X_train = X_train/255 
    
    # Prepare labels 
    values = np.array(train_df['breed'])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_train = onehot_encoded

    return X_train, y_train, label_encoder


def prepare_images(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 32, 32, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 32x32x3
        img = image.load_img("keras/dog_breed_identification/"+dataset+"/"+fig, target_size=(32, 32, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train


def preprocess_test_data (test_df):
	x_test = prepare_images(test_df, test_df.shape[0], "test")
	x_test /= 255
	return x_test


def build_model(X_train, y_train):
	
	model = Sequential()

	model.add (Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32,32,3)))
	model.add (Conv2D(32, (3,3), activation="relu"))
	model.add (MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	
	model.add (Conv2D(64, (3,3), padding="same", activation="relu"))
	model.add (Conv2D(64, (3,3), activation="relu"))
	model.add (MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(y_train.shape[1], activation='softmax'))
	return model

def main(argv=None):
	del argv  # Unused.

	announce()

	# Read data.
	train_df = pd.read_csv(DIR + "labels.csv")

	# preprocess.
	X_train, y_train, label_encoder  = preprocess_train_data (train_df)
	
	print ('X_train shape: {}, y_train shape {}'.format (X_train.shape,y_train.shape))

	# Build and compile a model.
	model = build_model(X_train, y_train)
	
	# Compile.
	model.compile(loss='categorical_crossentropy', 
		optimizer="adam", 
		metrics=['accuracy'])
	
	print(model.summary())

	# Train the model..
	history =model.fit(
		X_train,
		y_train,
		epochs=EPOCHS,
		batch_size=25,
		verbose=1,
		shuffle=True
		# verbose=2,
		# callbacks=[logger]
		)

	# plot the model's Accuracy. 
	plt.plot(history.history['acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.show()


	# Save the nueral network to a file.
	# 1. Save the network structure 
	model_structure = model.to_json()
	f = Path("keras/dog_breed_identification/model_structure.json")
	#f.open('w').write(model_structure)
	f.write_text(model_structure)

	# 2. Save the weights.
	model.save_weights("keras/dog_breed_identification/model_weights.h5")

	
def load_and_predict():

	train_df = pd.read_csv(DIR + "labels.csv")
	values = np.array(train_df['breed'])
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
    
	f = Path("keras/dog_breed_identification/model_structure.json")
	model_structure = f.read_text()
	model = model_from_json(model_structure)

	# load weights.
	model.load_weights("keras/dog_breed_identification/model_weights.h5")

	# predict
	test = os.listdir(TEST_DIR)
	col = ['Image']
	test_df = pd.DataFrame(test, columns=col)
	test_df['Id'] = ''
	X_test  = preprocess_test_data (test_df)
	predictions = model.predict(np.array(X_test), verbose=1)

	for i, pred in enumerate(predictions):
		test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
	print(test_df.head(20))


if __name__ == '__main__':
	if len(sys.argv) == 2 and sys.argv[1] == 'predict':
		load_and_predict()
	
	elif len(sys.argv) == 2 and sys.argv[1] == 'train':
		main()
	else:
		main()