import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

IMG_SIZE = (150, 150)

def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')  # Grayscale
    img = img.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1,150,150,1)
    return img_array, np.array(img)

def make_gradcam_heatmap(model, img_array, last_conv_layer_name="conv2"):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), predictions.numpy()[0][0]

def overlay_gradcam_on_image(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, image.shape[::-1])
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_color, 0.6, heatmap_color, 0.4, 0)
    return overlay
