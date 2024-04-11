import tensorflow as tf # library to build neural networks on graph data
from tensorflow import keras
from tensorflow import data as tf_data
import tensorflow_datasets as tfds # library of public datasets
import matplotlib.pyplot as plt # to visualize the data
import numpy as np

import ssl
import requests
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

# Some shortcuts for ease of use
endl = "\n"
div = "\n" + "-"*87
ttl = endl*2


print(ttl, "PROGRAM STARTED", endl)

#? Splitting Data 
train_ds, valid_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # dataset returned as (input, label), 
    # Input = images of cats or dogs, Label is 0 for cat, 1 for dog.
)
# After first run:
# Dataset cats_vs_dogs downloaded and prepared to 
# /Users/nishkaawasthi/tensorflow_datasets/cats_vs_dogs/4.0.1. 
# Subsequent calls will reuse this data.

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {valid_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")


#? Standardizing Data 
# Raw images have a variety of sizes. In addition, each pixel consists 
# of 3 integer values between 0 and 255 (RGB level values). This isn't 
# a great fit for feeding a neural network. 
# We need to do 2 things:
#       - Standardize to a fixed image size. We pick 150x150.
#       - Normalize pixel values between -1 and 1. We'll do this using 
#         a Normalization layer as part of the model itself.

# create a Resizing layer that resizes input images to 150 x 150 pixels
resize_fn = keras.layers.Resizing(150, 150) 

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
valid_ds = valid_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

#? Data Augmentation
# With limited data, data augmentation mimics realistic 'differences'
# in the data sets, like flips or rotations.
# This helps prevent overfitting

#& What about stretching and recoloring? B&W or saturation changes?

augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

#& Why only to training data? Why not to validation or test?


#? Batching that data + Prefetching
# By batching the data, you combine multiple augmented images into 
# each batch, which can then be processed in parallel by your neural 
# network. Prefetching further optimizes the loading speed by 
# overlapping data preprocessing and model execution, reducing 
# potential I/O and processing bottlenecks.

batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
valid_ds = valid_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()



#? LETS BUILD OUR MODEL
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

print(endl*3)
model.summary(show_trainable=True)

#? Train the Model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 2
print(endl, "Fitting the top layer of the model")
model.fit(train_ds, epochs=epochs, validation_data=valid_ds)

#? Fine Tuning
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.

base_model.trainable = True
model.summary(show_trainable=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 1
print("Fitting the end-to-end model")
model.fit(train_ds, epochs=epochs, validation_data=valid_ds)

#? Test Dataset Evaluation
print(endl, "Test dataset evaluation")
model.evaluate(test_ds)

print(ttl, "PROGRAM FINISHED", endl)
