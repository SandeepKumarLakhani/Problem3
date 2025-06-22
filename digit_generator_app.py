import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(784, activation='sigmoid'),
        tf.keras.layers.Reshape((28, 28))
    ])
    return model

generator = build_generator()
generator.load_weights("generator_weights.h5")

def generate_images(n):
    noise = tf.random.normal([n, 100])
    return generator(noise, training=False).numpy()

st.set_page_config(page_title="Digit Generator", layout="wide")
st.title("✍️ Handwritten Digit Generator (MNIST-Trained)")

n = st.slider("Number of digits to generate", 1, 10, 5)
if st.button("Generate"):
    images = generate_images(n)
    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
