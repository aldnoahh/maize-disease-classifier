# python librairies installation
#!pip install tensorflow opencv-python matplotlib split-folders matplotlib opencv-python spicy tf2onnx onnx


import numpy as np
import cv2 as cv
import os
import splitfolders
import matplotlib.pyplot as plt
import tf2onnx, onnx

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

data_folder = "MaizeClassification"
# split data in a new folder named data-split
splitfolders.ratio(data_folder, output="data-split", seed=1337, ratio=(0.7, 0.2, 0.1), group_prefix=None, move=False)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
# define classes name
class_names = os.listdir(data_folder)

# training data
train_generator = datagen.flow_from_directory(
    directory="data-split/train/",
    classes = class_names,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
)
# validation data
valid_generator = datagen.flow_from_directory(
    directory="data-split/val/",
    classes = class_names,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
)
# test data
test_generator = datagen.flow_from_directory(
    directory="data-split/test/",
    classes = class_names,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
)

# ResNet50 model
resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
for layer in resnet_50.layers:
    layer.trainable = False
# build the entire model
x = resnet_50.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(len(class_names), activation='softmax')(x)
model = Model(inputs = resnet_50.input, outputs = predictions)
# define training function
def trainModel(model, epochs, optimizer):
    batch_size = 32
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch_size)

# launch the training
model_history = trainModel(model = model, epochs = 30, optimizer = "Adam")

"""- Display **loss** curves:

- Display **accuracy** curves:
"""

loss_train_curve = model_history.history["loss"]
loss_val_curve = model_history.history["val_loss"]
plt.plot(loss_train_curve, label = "Train")
plt.plot(loss_val_curve, label = "Validation")
plt.legend(loc = 'upper right')
plt.title("Loss")
#plt.show()
plt.savefig('loss.png')


acc_train_curve = model_history.history["accuracy"]
acc_val_curve = model_history.history["val_accuracy"]
plt.plot(acc_train_curve, label = "Train")
plt.plot(acc_val_curve, label = "Validation")
plt.legend(loc = 'lower right')
plt.title("Accuracy")
#plt.show()
plt.savefig('accuracy.png')

test_loss, test_acc = model.evaluate(test_generator)
print("The test loss is: ", test_loss)
print("The best accuracy is: ", test_acc*100)

# img = tf.keras.preprocessing.image.load_img('/content/data-split/test/Foreign/1652253420048.png_485_2410_280_258_174.png', target_size=(224, 224))
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = np.array([img_array])
# #img
# # generate predictions for samples
# predictions = model.predict(img_array)
# print(predictions)
# # generate argmax for predictions
# class_id = np.argmax(predictions, axis = 1)
# print(class_id)
# # transform classes number into classes name
# class_names[class_id.item()]

model.save('model.h5')

# Load the Keras model
model = keras.models.load_model("model.h5")

# Convert the model to ONNX
#onnx_model = tf2onnx.convert_keras(model, target_opset=13)
onnx_model,_ = tf2onnx.convert.from_keras(model)

# Save the ONNX model
onnx.save_model(onnx_model, "model.onnx")
#model.summary()


