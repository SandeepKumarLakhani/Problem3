import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100
EPOCHS = 50

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator = build_generator()
discriminator = build_discriminator()
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_images = generator(noise, training=True)
        real_out = discriminator(images, training=True)
        fake_out = discriminator(gen_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_out), fake_out)
        disc_loss = cross_entropy(tf.ones_like(real_out), real_out) +                     cross_entropy(tf.zeros_like(fake_out), fake_out)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            train_step(batch)
        print(f'Epoch {epoch+1}/{epochs} done')
    generator.save_weights("generator_weights.h5")
    print("✅ Generator weights saved to generator_weights.h5")

if __name__ == "__main__":
    train(dataset, EPOCHS)
