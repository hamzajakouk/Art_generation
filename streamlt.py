import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import pprint
from public_tests import *
from style_content import *
# Import other necessary functions from your code

# Main Streamlit app
def main():
    st.title("Neural Style Transfer App")

    # Upload images
    content_image = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
    style_image = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

    if content_image and style_image:
        # Load and preprocess uploaded images
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)

        # Display uploaded images
        st.image([content_image, style_image], caption=["Content Image", "Style Image"], width=300)

        # Perform style transfer
        images = perform_style_transfer(content_image, style_image)  # Call your style transfer function here

        # Display generated image
        
        st.image(images, caption=f"Generated Image - Epoch", width=300)

        # Save the generated image as a file
        

# Helper function to perform style transfer
def perform_style_transfer(content_image, style_image):
    # Convert PIL images to numpy arrays
    tf.random.set_seed(272) # DO NOT CHANGE THIS VALUE
    pp = pprint.PrettyPrinter(indent=4)
    img_size = 400
    vgg = tf.keras.applications.VGG19(include_top=False,
                                    input_shape=(img_size, img_size, 3),
                                    weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False
    
       
        # Perform your style transfer process here
        # You need to adapt and integrate the necessary parts of your code

        # Convert the generated image back to a PIL image
    STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]
    
    
    content_image = np.array(content_image.resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    style_image =  np.array(style_image.resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
    style_image = style_image[:, :, :, :3]
    generated = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated), -0.25, 0.25)
    generated = tf.add(generated, noise)
    generated = tf.clip_by_value(generated, clip_value_min=0.0, clip_value_max=1.0)
    
    content_layer = [('block5_conv4', 1)]
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs(style_image)     # Style encoder
    # Assign the content image to be the input of the VGG model.  
    # Set a_C to be the hidden layer activation from the layer we have selected
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input.
    a_G = vgg_model_outputs(generated)

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    # Assign the input of the model to be the "style" image 
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    # Compute the style cost
    J_style = compute_style_cost(a_S, a_G)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    @tf.function()
    def train_step(generated):
        with tf.GradientTape() as tape:
            # In this function you must use the precomputed encoded images a_S and a_C
            
            ### START CODE HERE
            
            # Compute a_G as the vgg_model_outputs for the current generated image
            #(1 line)
            content_layer = [('block5_conv4', 1)]
            vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
            a_G = vgg_model_outputs(generated)
            
            # Compute the style cost
            #(1 line)
            J_style = compute_style_cost(a_S , a_G)

            #(2 lines)
            # Compute the content cost
            J_content = compute_content_cost(a_C , a_G)
            # Compute the total cost
            J = total_cost( J_content, J_style, alpha = 10, beta =40 )
            
            ### END CODE HERE
            
        grad = tape.gradient(J, generated)

        optimizer.apply_gradients([(grad, generated)])
        generated.assign(clip_0_1(generated))
        # For grading purposes
        return J
    generated = tf.Variable(generated)
    epochs = 600

    for i in range(epochs):
        train_step(generated)

        if i % 600 == 0:  # Display and save images at specific epochs
            image = tensor_to_image(generated)
            


    return image
    
if __name__ == "__main__":
    main()
