# -*- coding: utf-8 -*-
"""GAN on MNIST with checkpoints + image saving"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape, Dropout
import numpy as np


# -----------------------
# Load dataset
# -----------------------
from tensorflow.keras.datasets.mnist import load_data
(X_train, _), (_, _) = load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_train = (X_train - 127.5) / 127.5   # Normalize to [-1,1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True).repeat()

steps_per_epoch = X_train.shape[0] // BATCH_SIZE

# -----------------------
# Generator
# -----------------------
def make_generator():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# -----------------------
# Discriminator
# -----------------------
def make_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))
    return model

# -----------------------
# Models + Loss + Optims
# -----------------------
generator = make_generator()
discriminator = make_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# -----------------------
# Checkpoints with Manager
# -----------------------
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)

# Keep only the latest 3 checkpoints
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restore latest if exists
if manager.latest_checkpoint:
    print("Restoring from", manager.latest_checkpoint)
    checkpoint.restore(manager.latest_checkpoint)

# -----------------------
# Training Loop
# -----------------------
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, 0] * 127.5 + 127.5).numpy().astype("uint8"), cmap='gray')
        plt.axis('off')

    # Save images instead of just showing
    os.makedirs("generated_images", exist_ok=True)
    plt.savefig(f"generated_images/epoch_{epoch:03d}.png")
    plt.close(fig)

def train(dataset, epochs):
    for epoch in range(1, epochs+1):
        for step, image_batch in enumerate(dataset.take(steps_per_epoch)):
            gen_loss, disc_loss = train_step(image_batch)

        print(f"Epoch {epoch}/{epochs} | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f}")
        generate_and_save_images(generator, epoch, seed)

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_path = manager.save()
            print(f"Checkpoint saved at {save_path}")

# -----------------------
# Run training
# -----------------------
train(train_dataset, EPOCHS)



