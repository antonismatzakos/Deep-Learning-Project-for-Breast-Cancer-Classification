import numpy as np
import keras
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.datasets import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt



vgg_model = tensorflow.keras.applications.vgg16.VGG16()


type(vgg_model)



#to model den htan tupou sequential opote to pernaw se ena adeio modelo typou sequential

model = Sequential()
for layer in vgg_model.layers[:-1]:
    model.add(layer)

#afairw to teleutaio layer giati thelw duo kathgories eikonwn(dn mporw na to kanw me to pop)


#make all layers non trainable gia na apofugw oso ginetai to overfitting
for layer in model.layers:
    layer.trainable = False



# Add Dropout


# 3rd Fully Connected Layer

model.add(Dense(2, activation='softmax'))


model.summary()
train_path="C:/Users/John Matzakos/Desktop/Antonis/Data NEW/Train"
valid_path="C:/Users/John Matzakos/Desktop/Antonis/Data NEW/Valid"
test_path="C:/Users/John Matzakos/Desktop/Antonis/Data NEW/Test"

#pairnw tis eikones apto directory kai me to image data geerator paragw batches apo normalized data
train_batches= ImageDataGenerator(  horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2).flow_from_directory(train_path, target_size=(224,224), classes=['benign','malignant'], batch_size=22)
valid_batches= ImageDataGenerator( horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2).flow_from_directory(valid_path, target_size=(224,224), classes=['benign','malignant'], batch_size=10)
test_batches= ImageDataGenerator( horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2).flow_from_directory(test_path, target_size=(224,224), classes=['benign','malignant'], batch_size=15, shuffle= False)

#test_imgs, test_labels = next(test_batches)



model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_batches, steps_per_epoch=25, validation_data=valid_batches, validation_steps=25, epochs=200, verbose=2)
#Results = model.predict_generator(test_batches)
#print(Results)


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





