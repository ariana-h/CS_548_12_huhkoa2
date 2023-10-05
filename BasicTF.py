import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg19 import VGG19
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
from sklearn.model_selection import train_test_split


def load_catdog_filenames(basedir):
    all_filenames = os.listdir(basedir)    
    
    train_list, test_list = train_test_split(all_filenames,
                                             train_size=0.70,
                                             random_state=42)
    
    train_ds = tf.data.Dataset.from_tensor_slices(train_list)
    
    test_ds = tf.data.Dataset.from_tensor_slices(test_list)
    
    def load_image(x):
        if tf.strings.regex_full_match(x, "dog.*"):
            label = 0
        else:
            label = 1
        rawdata = tf.io.read_file(basedir + "/" + x)
        image = tf.io.decode_jpeg(rawdata)
        image = tf.image.convert_image_dtype(image, tf.float32)
        #tf.print("Image:", tf.shape(image))
        image = tf.image.resize(image, (32,32))
        return image, label
    
    train_ds = train_ds.map(load_image)
    test_ds = test_ds.map(load_image)
        
    
    train_iter = iter(train_ds)
    
    for _ in range(5):
        x = next(train_iter)
        image = x[0]
        label = x[1]
        image = image.numpy()
        label = label.numpy()
        print(image.shape, label)
           
    return train_ds, test_ds

def main():
    
    train_ds, test_ds = load_catdog_filenames("../catdog")
    
    
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(2))
    
    
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    '''
    
    my_input = Input(shape=x_train.shape[1:])
    x = Conv2D(32, kernel_size=3,
                    padding="same", 
                    activation="relu")(my_input)
    
    x = Conv2D(32, kernel_size=3, 
               padding="same", 
               activation="relu")(x)
    
    x = MaxPooling2D(2)(x)
    
    alt_x = Dense(64)(x)
    
    x = Conv2D(64, kernel_size=3, 
               padding="same", 
               activation="relu")(x)
    
    x =Conv2D(64, kernel_size=3, 
              padding="same", 
              activation="relu")(x)
    
    x = Add()([x,alt_x])
    
    x = MaxPooling2D(2)(x)
    
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    my_output = Dense(2, activation="softmax")(x)
    
    model = Model(inputs=my_input, outputs=my_output)
    
    
    base_model = VGG19(weights = "imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
        
    true_input = Input(shape=image_shape)
    resized = Lambda(input_shape=image_shape,
                     function=lambda images: 
                         tf.image.resize(images,[224,224]))(true_input)
    x = base_model(resized)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
    
    model = Model(inputs=true_input, outputs=x)
    
    
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    total_epoch_cnt = 5
    
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=["accuracy"])
    
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", 
                                                 histogram_freq=1)
    
    #model.fit(x_train, y_train, batch_size=32, epochs=5, callbacks=[tb_callback])
    
    tb_callback = tf.keras.callbacks.TensorBoard(
                    log_dir="logs",
                    histogram_freq=1)
        
    #model.fit(x_train, y_train, batch_size=32, epochs=5,
    #          callbacks=[tb_callback])
    model.fit(train_ds, epochs=5,
              validation_data=test_ds,
              callbacks=[tb_callback])
    
    train_scores = model.evaluate(train_ds, 
                                  batch_size=128)
    test_scores = model.evaluate(test_ds,
                                 batch_size=128)
    
    print("TRAIN:", train_scores)
    print("TEST:", test_scores)


if __name__=="__main__":
    main()