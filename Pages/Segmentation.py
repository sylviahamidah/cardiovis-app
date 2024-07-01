import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.initializers import glorot_uniform, glorot_normal, orthogonal, he_normal, lecun_normal
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')

from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Conv2D, Dropout
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import model_from_json
import cv2
import gdown
import os

#-------------------------------------------------------------------------------------------------------------
@register_keras_serializable(package="Custom", name="resize0")
def resize0(x):
    return tf.image.resize(x, size=(16, 16), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize1")
def resize1(x):
    return tf.image.resize(x, size=(32, 32), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize2")
def resize2(x):
    return tf.image.resize(x, size=(64, 64), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize3")
def resize3(x):
    return tf.image.resize(x, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize4")
def resize4(x):
    return tf.image.resize(x, size=(256, 256), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize0_output")
def resize0_output(input_shape = (None, 256, 256, 3)):
    return (input_shape[0], 16, 16, 1024)

@register_keras_serializable(package="Custom", name="resize1_output")
def resize1_output(input_shape = (None, 256, 256, 3)):
    return (input_shape[0], 32, 32, 256)

@register_keras_serializable(package="Custom", name="resize2_output")
def resize2_output(input_shape = (None, 256, 256, 3)):
    return (input_shape[0], 64, 64, 128)

@register_keras_serializable(package="Custom", name="resize3_output")
def resize3_output(input_shape = (None, 256, 256, 3)):
    return (input_shape[0], 128, 128, 64)

@register_keras_serializable(package="Custom", name="resize4_output")
def resize4_output(input_shape = (None, 256, 256, 3)):
    return (input_shape[0], 256, 256, 64)
    
@register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    dice = tf.keras.losses.Dice()
    return bce(y_true, y_pred) + dice(y_true, y_pred)

def identity_block(X, f, filters, stage, block):
    
    # tuning
    initializer = glorot_uniform(seed=0)
    function = 'relu'

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = initializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation(function)(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = initializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation(function)(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = initializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation(function)(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # tuning
    initializer = glorot_uniform(seed=0)
    function = 'relu'
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = initializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation(function)(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = initializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation(function)(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = initializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = initializer)(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation(function)(X)

    return X

def AlbuNet(input_shape = (256, 256, 3), classes = 1):
    
    X_input = Input(input_shape)
    
    # Zero-Padding
    #X_input = ZeroPadding2D((3, 3))(Input(input_shape))
    
    # Stage 0
    X1 = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    #X1 = BatchNormalization(axis = 3, name = 'bn_conv1')(X1)
    #X1 = Activation('relu')(X1)
    X1 = MaxPooling2D((3, 3), strides=(1, 1))(X1)
    
    # Stage 1
    X2 = convolutional_block(X1, f = 3, filters = [64, 64, 64], stage = 2, block='a', s = 1)
    X2 = MaxPooling2D((3, 3), strides=(1, 1))(X2)
    X2 = identity_block(X2, 3, [64, 64, 64], stage=2, block='b')
    X2 = identity_block(X2, 3, [64, 64, 64], stage=2, block='c')
    
    ### START CODE HERE ###
    # Stage 2 (≈4 lines)
    X3 = convolutional_block(X2, f = 3, filters = [128, 128, 128], stage = 3, block='a', s = 2)
    X3 = MaxPooling2D((3, 3), strides=(1, 1))(X3)
    X3 = identity_block(X3, 3, [128, 128, 128], stage=3, block='b')
    X3 = identity_block(X3, 3, [128, 128, 128], stage=3, block='c')
    X3 = identity_block(X3, 3, [128, 128, 128], stage=3, block='d')
    
    # Stage 3 (≈6 lines)
    X4 = convolutional_block(X3, f = 3, filters = [256, 256, 256], stage = 4, block='a', s = 2)
    X4 = MaxPooling2D((3, 3), strides=(1, 1))(X4)
    X4 = identity_block(X4, 3, [256, 256, 256], stage=4, block='b')
    X4 = identity_block(X4, 3, [256, 256, 256], stage=4, block='c')
    X4 = identity_block(X4, 3, [256, 256, 256], stage=4, block='d')
    X4 = identity_block(X4, 3, [256, 256, 256], stage=4, block='e')
    X4 = identity_block(X4, 3, [256, 256, 256], stage=4, block='f')
    
    # Stage 4 (≈3 lines)
    X5 = convolutional_block(X4, f = 3, filters = [512, 512, 512], stage = 5, block='a', s = 2)
    X5 = MaxPooling2D((3, 3), strides=(1, 1))(X5)
    X5 = identity_block(X5, 3, [512, 512, 512], stage=5, block='b')
    X5 = identity_block(X5, 3, [512, 512, 512], stage=5, block='c')
    
    # Bottleneck
    X_bottleneck = Conv2D(1024, (3, 3), strides = (1,1), padding = 'valid', name = "bottle_neck", kernel_initializer = glorot_uniform(seed=0))(X5)
    resize_0 = Lambda(resize0, output_shape=resize0_output)
    X_bottleneck = resize_0(X_bottleneck)
    
    
    # Decoder 1
    X_dec1 = layers.UpSampling2D(size=(2, 2))(X_bottleneck)
    #resize1 = Lambda(lambda x: tf.image.resize(x, size=(32, 32), method=tf.image.ResizeMethod.BILINEAR), output_shape=(32, 32, 256))
    resize_1 = Lambda(resize1, output_shape=resize1_output)
    X4_resized = resize_1(X4)
    X_dec1 = layers.concatenate([X_dec1, X4_resized], axis=-1)
    X_dec1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(X_dec1)
    X_dec1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(X_dec1)
    X_dec1 = BatchNormalization(axis = 3)(X_dec1)
    X_dec1 = Activation('relu')(X_dec1)
    
    # Decoder 2
    X_dec2 = layers.UpSampling2D(size=(2, 2))(X_dec1)
    #resize2 = Lambda(lambda x: tf.image.resize(x, size=(64, 64), method=tf.image.ResizeMethod.BILINEAR), output_shape=(64, 64, 128))
    resize_2 = Lambda(resize2, output_shape=resize2_output)
    X3_resized = resize_2(X3)
    X_dec2 = layers.concatenate([X_dec2, X3_resized], axis=-1)
    X_dec2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(X_dec2)
    X_dec2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(X_dec2)
    X_dec2 = BatchNormalization(axis = 3)(X_dec2)
    X_dec2 = Activation('relu')(X_dec2)
    
    # Decoder 3
    X_dec3 = layers.UpSampling2D(size=(2, 2))(X_dec2)
    #resize3 = Lambda(lambda x: tf.image.resize(x, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR), output_shape=(128, 128, 64))
    resize_3 = Lambda(resize3, output_shape=resize3_output)
    X2_resized = resize_3(X2)
    X_dec3 = layers.concatenate([X_dec3, X2_resized], axis=-1)
    X_dec3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(X_dec3)
    X_dec3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(X_dec3)
    X_dec3 = BatchNormalization(axis = 3)(X_dec3)
    X_dec3 = Activation('relu')(X_dec3)
    
    # Decoder 4
    X_dec4 = layers.UpSampling2D(size=(2, 2))(X_dec3)
    #resize4 = Lambda(lambda x: tf.image.resize(x, size=(256, 256), method=tf.image.ResizeMethod.BILINEAR), output_shape=(256, 256, 64))
    resize_4 = Lambda(resize4, output_shape=resize4_output)
    X1_resized = resize_4(X1)
    X_dec4 = layers.concatenate([X_dec4, X1_resized], axis=-1)
    X_dec4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(X_dec4)
    X_dec4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(X_dec4)
    X_dec4 = BatchNormalization(axis = 3)(X_dec4)
    X_dec4 = Activation('relu')(X_dec4)
    
    # output layer
    X_output = Conv2D(1, (1, 1), padding = 'valid', name = "conv_output")(X_dec4)
    X_output = Activation('sigmoid')(X_output)
    
    # Create model
    model = Model(inputs = X_input, outputs = X_output, name='AlbuNet')
    
    return model
#-------------------------------------------------------------------------------------------------------------
def load_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=-1)  # Menambahkan dimensi saluran
    img = tf.image.grayscale_to_rgb(img)
    img = img / 255.0
    return img

def load_model(model_path, weight_path, image):
    with open(model_path, 'r') as f:
        model_json = f.read()

    custom_objects = {
        "resize0" : resize0,
        "resize1" : resize1,
        "resize2" : resize2,
        "resize3" : resize3,
        "resize4" : resize4,
        "resize0_output" : resize0_output,
        "resize1_output" : resize1_output,
        "resize2_output" : resize2_output,
        "resize3_output" : resize3_output,
        "resize4_output" : resize4_output,
        "custom_loss" : custom_loss,
        "Lambda": Lambda
        }

    model = model_from_json(model_json, custom_objects=custom_objects) #, safe_mode=False)
    model.load_weights(weight_path)
    segmentation = model.predict(tf.expand_dims(image, axis=0))[0]
    return segmentation

#-------------------------------------------------------------------------------------------------------------
def merge_files(output_path, input_paths):
    with open(output_path, 'wb') as output_file:
        for input_path in input_paths:
            with open(input_path, 'rb') as input_file:
                while chunk := input_file.read(1024 * 1024):  # Membaca per 1 MB
                    output_file.write(chunk)


# Daftar file part yang akan digabungkan
input_path1 = [f'model_split\heart_fix.weights.h5_part_{i}' for i in range(0, 17)]
input_path2 = [f'model_split\left_fix.weights.h5_part_{i}' for i in range(0, 17)]
input_path3 = [f'model_split\Right_fix.weights.h5_part_{i}' for i in range(0, 17)]

# Nama file hasil penggabungan
weight_path1 = 'model_heart.weights.h5'
weight_path2 ='model_left.weights.h5'
weight_path3 = 'model_right.weights.h5'

# Memanggil fungsi untuk menggabungkan file
merge_files(weight_path1, input_path1)
merge_files(weight_path2, input_path2)
merge_files(weight_path3, input_path3)

#-------------------------------------------------------------------------------------------------------------

st.title(":two: Image Segmentation")
col1, col2 = st.columns([0.3, 0.7], gap='medium')

with col1:
    st.markdown('#### Area Options')
    st.write("Select the area to analyze:")
    heart = st.checkbox("Heart")
    left_lung = st.checkbox("Left Lung")
    right_lung = st.checkbox("Right Lung")
    apply = st.button("Apply", type="secondary")
    st.write("OR")
    run_all = st.button("Run All", type="secondary")
    
    if run_all:
        left_lung = True
        right_lung = True
        heart = True
        apply = True

with col2:
    st.markdown('#### Output Images')
    tab1, tab2, tab3, tab4 = st.tabs(["Input Images", "Heart Segmentation", "Left Lung Segmentation", "Right Lung Segmentation"])
    
    with tab1:
        if st.session_state.get('next_step', False):
            if 'result' in st.session_state:
                input_image = st.session_state['result']
                st.image(input_image, width=300)
        
    if apply:
        if heart:
            model_path1 = "E:\TA\Code\Streamlit\heart_fix.json"
            #weight_path1 = "E:\TA\Code\Streamlit\heart_fix.weights.h5"
            segment_input = load_image(input_image)
            heart_output = load_model(model_path1, weight_path1, segment_input)
            st.session_state['heart_output'] = heart_output
            with tab2:
                st.image(heart_output, width=300)
            
        if left_lung:
            model_path2 = "E:\TA\Code\Streamlit\left_fix.json"
            #weight_path2 = "E:\TA\Code\Streamlit\left_fix.weights.h5"
            segment_input = load_image(input_image)
            left_output = load_model(model_path2, weight_path2, segment_input)
            st.session_state['left_output'] = left_output
            with tab3:
                st.image(left_output, width=300)
            
        if right_lung:
            model_path3 = "E:\TA\Code\Streamlit\Right_fix.json"
            #weight_path3 = "E:\TA\Code\Streamlit\Right_fix.weights.h5"
            segment_input = load_image(input_image)
            right_output = load_model(model_path3, weight_path3, segment_input)
            st.session_state['right_output'] = right_output
            with tab4:
                st.image(right_output, width=300)
