import numpy as np
import keras
import tensorflow
from keras import preprocessing

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.datasets import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from tensorflow.keras.applications import imagenet_utils

mobile = tensorflow.keras.applications.mobilenet.MobileNet()

#preparing image paths
train_path="C:/Users/John Matzakos/Desktop/Antonis/Data NEW/Train"
valid_path="C:/Users/John Matzakos/Desktop/Antonis/Data NEW/Valid"
test_path="C:/Users/John Matzakos/Desktop/Antonis/Data NEW/Test"

#pairnw tis eikones apto directory kai me to image data generator paragw batches apo normalized data
train_batches= ImageDataGenerator(featurewise_center=True,
    samplewise_center=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,


    shear_range=0.2,

    horizontal_flip=True,
    vertical_flip=True,preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(224,224), classes=['benign','malignant'], batch_size=26)
valid_batches= ImageDataGenerator(featurewise_center=True,
    samplewise_center=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,


    shear_range=0.2,

    horizontal_flip=True,
    vertical_flip=True,preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224,224), classes=['benign','malignant'], batch_size=5)
#shuffle= false gia na mhn ginei shuffle sto test datasetm, exei na kanei me to confusion matrix
test_batches= ImageDataGenerator(featurewise_center=True,
    samplewise_center=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,


    shear_range=0.2,

    horizontal_flip=True,
    vertical_flip=True,preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(224,224), classes=['benign','malignant'], batch_size=15, shuffle=False)


"""
pairnw to output apo to 6 mexri to teleutaio layer sto original mobilenet modelo kai to kataxwrw ston x
kanontas fine tuning to modelo krataw th suntriptikh pleiopshfia twn layers (88 layers) kai afairw kapoia giati eksuphretei to skopo mou
krataw mexri kai to global average pooling layer
"""
x = mobile.layers[-6].output


"""
katopin xrhsimopoioume ena dense layer predictions gia na kathorisoume oti tha exoume diaxwropoihsh se duo kathgories benign kai malignant
to modelo to ftiaksa me tupou model keras functional api ki oxi sequential opws to vgg16
"""
predictions = Dense(2, activation = 'softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)

"""
gia na xtisoume to neo modelo dhmiourgoume ena paradeigma tou modelou tou original kai dinoume sugkekrimena inputs tou idiou tou mobilenet 
kai meta kathorizoume oti output tou modelou tha einai to layer prediction me duo kathgoriopoihseis
ara ousiastika xtizw ena modelo idio me to mobile net mono pou dn exei ta 5 teleutaia original layers me th diafora enos kainougiou teleutaiou layer gia na eskuphphretw to sugkekrimeno skopo mou 
"""

model.summary()





model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
for layer in model.layers:
    layer.trainable = False
history=model.fit_generator(train_batches, steps_per_epoch=25, validation_data=valid_batches, validation_steps=25, epochs=200, verbose=2)



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_batches, 1474 // 15+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_batches.classes, y_pred))
print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(test_batches.classes, y_pred, target_names=target_names))