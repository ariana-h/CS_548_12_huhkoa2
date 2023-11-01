#using code from https://www.kaggle.com/code/tapliershiru/pix2pix-image-to-image-translation-tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Concatenate,
                                     Conv2D,
                                     Conv2DTranspose,
                                     Input,
                                     BatchNormalization,
                                     Dropout,
                                     ReLU,
                                     LeakyReLU,
                                     ZeroPadding2D
                                     )
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import glob
import matplotlib.pyplot as plt
import os
import time 
from IPython import display


PATH_TO_DATA= "/media/huhkoa2/Samsung USB/VSCode/VSCode-linux-x64/data/facade_dataset"

# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

NUM_OF_DATA = len(glob.glob(os.path.join(PATH_TO_DATA, "trainA/*")))
NUM_OF_DATA_TEST = len(glob.glob(os.path.join(PATH_TO_DATA, "testA/*")))
NUM_OF_DATA, NUM_OF_DATA_TEST


#LOAD DATA
def load_images(index: int, is_test=False):
    if is_test:
        path_real = f"testA/{index}.jpg"
        path_input = f"testB/{index}.jpg"
    else:
        path_real = f"trainA/{index}_A.jpg"
        path_input = f"trainB/{index}_B.jpg"
    train_real_path = os.path.join(PATH_TO_DATA, path_real)
    train_input_path = os.path.join(PATH_TO_DATA, path_input)
    
    train_real = tf.io.read_file(train_real_path)
    train_real = tf.io.decode_jpeg(train_real)
    
    train_input = tf.io.read_file(train_input_path)
    train_input = tf.io.decode_jpeg(train_input)
    return train_input, train_real

'''
#TEST LOADING IMAGES - they show up in separate windows
single_input, single_real = load_images(1)
plt.figure()
plt.imshow(single_real)
plt.figure()
plt.imshow(single_input)
#according to https://stackoverflow.com/questions/34742842/use-imshow-of-matplotlib-pylab-with-a-python-ide need plt.show() once in IDE
#plt.show() 
'''


#AUGMENTATION METHODS
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], 
                             axis=0)
    cropped_image = tf.image.random_crop(stacked_image, 
                                         size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (tf.cast(input_image, dtype=tf.float32) / 127.5) - 1
    real_image = (tf.cast(real_image, dtype=tf.float32) / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

'''
#TEST AUG METHODS - 4 images show up in grid (all different)
plt.figure(figsize=(6, 6))
for i in range(4):
    rj_input, rj_real = random_jitter(single_input, single_real)
    plt.subplot(2, 2, i + 1)
    plt.imshow(rj_input)
    #plt.imshow(rj_real)
    plt.axis('off')
plt.show()
'''

#DEFINE GENERATOR
def data_generator(is_test=False):
    if is_test:
        max_number = NUM_OF_DATA_TEST
    else:
        max_number = NUM_OF_DATA
    
    counter = 1
    while True:
        input_image, real_image = load_images(counter, is_test=is_test)
        yield input_image, real_image
        
        counter += 1
        if counter > max_number:
            counter = 1
        

def load_image_train(input_image, real_image):
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(input_image, real_image):
    input_image, real_image = resize(
        input_image, real_image,
        IMG_HEIGHT, IMG_WIDTH
    )
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image



#CREATING DATASET | TF PIPELINE
base_train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(is_test=False),
    output_signature=(
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS), dtype=tf.uint8),
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS), dtype=tf.uint8))
)
train_dataset = base_train_dataset.map(
    load_image_train, num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.shuffle(NUM_OF_DATA)
train_dataset = train_dataset.batch(BATCH_SIZE)



test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(is_test=True),
    output_signature=(
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS), dtype=tf.uint8),
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS), dtype=tf.uint8))
)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


#TESTING PIPELINE
input_image, real_image = next(iter(test_dataset))
'''
plt.figure()
plt.imshow((input_image[0]+1)/2)
plt.figure()
plt.imshow((real_image[0]+1)/2)
#plt.show()
'''

#ENCODER
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential() 
    result.add(Conv2D(filters, size, 
                      strides=2, padding='same',
                      kernel_initializer=initializer, 
                      use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return  result

down_model = downsample(3, 4)
down_result = down_model(input_image) 
print (down_result.shape)


#DECODER
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = Sequential() 
    result.add(Conv2DTranspose(filters, size, 
                               strides=2, padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    result.add(BatchNormalization())

    if apply_dropout:
          result.add(Dropout(0.5))
          
    result.add(ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)


#GENERATOR MODEL
def Generator():
    inputs = Input(shape=[256, 256, 3])
    
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(
        OUTPUT_CHANNELS, 4, strides=2,
        padding='same',kernel_initializer=initializer,
        activation='tanh'
    )  # (batch_size, 256, 256, 3)
    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
    
    x = last(x)
    
    return Model(inputs=inputs, outputs=x)
    

generator = Generator()
gen_output = generator(input_image)
'''
plt.figure()
plt.imshow((input_image[0]+1)/2)
plt.figure()
plt.imshow((gen_output[0]+1)/2)
#plt.show()
'''

#GENERATOR LOSS
LAMBDA = 100
loss_object = BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

#DISCRIMINATOR
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    
    input = Input(shape=[256, 256, 3], name='input_image')
    target = Input(shape=[256, 256, 3], name='target_image')
    
    #x = tf.keras.layers.Concatenate([input_image, real_image])  # (batch_size, 256, 256, channels*2)
    x = tf.keras.layers.concatenate([input, target]) # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = Conv2D(512, 4, strides=1, 
                  kernel_initializer=initializer, use_bias=False
        ) (zero_pad1)# (batch_size, 31, 31, 512)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = Conv2D(1, 4, strides=1,
                  kernel_initializer=initializer
        ) (zero_pad2)  # (batch_size, 30, 30, 1)
    
    return Model(inputs=[input, target], outputs=last)
    
  
discriminator = Discriminator()
disc_out = discriminator([input_image, gen_output], training=False)
''' 
plt.figure()
plt.imshow(disc_out[0])
plt.colorbar()
#plt.show()
''' 

#DISCRIMINATOR LOSS
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

def main():
    #OPTIMIZERS
    generator_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = Adam(2e-4, beta_1=0.5)


    #CHECKPOINTS
    checkpoint_dir = 'Assign02/training_checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    #TEST TOOLS
    def generate_images(model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
    
    #TESTING
    for example_input, example_target in test_dataset.take(1):
        generate_images(generator, example_input, example_target)
      
      
    #LOG FILE
    import datetime
    log_dir="Assign02/logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )


    #STEP FUNCTION | SINGLE TRAINING STEP
    @tf.function
    def train_step(input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, 
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, 
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        

    #TRAINING
    def fit(train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                generate_images(generator, example_input, example_target)
                print(f"Step: {step//1000}k")

            train_step(input_image, target, step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)


            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            
    fit(train_dataset, test_dataset, steps=40000)

if __name__== "__main__":
    main()
