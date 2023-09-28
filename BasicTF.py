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
        lable = label.numpy()
        #x = x.numpy()
        #print(x.shape)
        
    exit(1)
    
    return train_list, test_list




def main():
    
    train_list, test_list = load_catdog_filenames("../catdog")
    print(train_list)
    
    print("HELLO")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("Image type:", x_train.dtype)

    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    def preprocess_images(x):
        x = x.astype("float32")
        x /= 255.0
        if len(x.shape) <= 3:
            x = np.expand_dims(x, axis=-1)  
        return x
    
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    
    print("x_train AFTER:", x_train.shape)
    print("x_test AFTER:", x_test.shape)
    print("Image type AFTER:", x_train.dtype)  
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    
    train_cnt = train_ds.cardinality()
    print("Number of training samples:", train_cnt)
    train_ds = train_ds.shuffle(train_cnt)
    
    train_ds = train_ds.batch(64)
    
    test_ds = test_ds.batch(64)
    
    #for image,label in train_ds:
    #    print("IMAGE:", image.numpy().shape)
    #    print("LABEL:", label.numpy().shape)
        
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
    
    model.summary()
    
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    
    model.fit(x_train, y_train, batch_size=32, epochs=5)
    
    train_scores = model.evaulate(x_train, y_train, 
                                  batch_size=128)
    test_scores = model.evaluate(x_test, y_test, 
                                 batch_size=128)
    
    print("TRAIN:", train_scores)
    print("TEST:", test_scores)
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
    my_output = Dense(10, activation="softmax")(x)
    
    model = Model(inputs=my_input, outputs=my_output)
    
    
    base_model = VGG19(weights = "imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    
    true_input = Input(shape=x_train.shape[1:])
    resized = Lambda(input_shape=x_train.shape[1:],
                     function=lambda images:tf.image.resize(images, [224,224]))(true_input)
    
    x=base_model(resized)
    x = Flatten()(x)
    x = Dense(1024,activation="relu")(x)
    x = Dense(1024,activation="relu")(x)
    x = Dense(10,activation="softmax")(x)
    
    model = Model(inputs=true_input, outputs=x)
    
    
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", 
                                                 histogram_freq=1)
    
    #model.fit(x_train, y_train, batch_size=32, epochs=5, callbacks=[tb_callback])
    
    model.fit(train_ds, epochs=5, validation_data=test_ds, callbacks=[tb_callback])
    
    
    train_scores = model.evaluate(x_train, y_train, 
                                  batch_size=128)
    test_scores = model.evaluate(x_test, y_test, 
                                 batch_size=128)
    
    print("TRAIN:", train_scores)
    print("TEST:", test_scores)


if __name__=="__main__":
    main()