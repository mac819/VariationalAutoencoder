import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Encoder
def encoder(inputs):
    # Block-1
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", name="conv_1")(inputs)
    x = layers.BatchNormalization(name="bn_enc_1")(x)
    x = layers.LeakyReLU(name="lrelu_enc_1")(x)
    
    
    # Block-2
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_enc_2")(x)
    x = layers.LeakyReLU(name="lrelu_enc_2")(x)
    
    # Block-3
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", name="conv_3")(x)
    x = layers.BatchNormalization(name="bn_enc_3")(x)
    x = layers.LeakyReLU(name="lrelu_enc_3")(x)
    
    # Block-4
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same", name="conv_4")(x)
    x = layers.BatchNormalization(name="bn_enc_4")(x)
    x = layers.LeakyReLU(name="lrelu_enc_4")(x)
    
    # Final Block
    flatten = layers.Flatten()(x)
    mean = layers.Dense(2, name="mean")(flatten)
    log_var = layers.Dense(2, name="log_var")(flatten)
    
    
    # Sampling Block
    latent_vec = layers.Lambda(sampling_reparameterization, name="encoder_output")([mean, log_var])
    
    return latent_vec, mean, log_var



# Sampling Layer
def sampling_reparameterization(distribution_params):
    mean, log_var = distribution_params
    epsilon = tf.random.normal(shape=tf.shape(mean), mean=0., stddev=1.)
    z = mean + tf.exp(log_var / 2) * epsilon
    return z


# Decoder
def decoder(latent_vec):
    
    x = layers.Dense(262144, name='dense_1')(latent_vec)
    x = layers.Reshape((64, 64, 64), name='Reshape_Layer')(x)
   
    # Block-1
    x = layers.Conv2DTranspose(64, 3, strides= 1, padding='same',name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_dec_1')(x)
    x = layers.LeakyReLU(name='lrelu_dec_1')(x)
  
    # Block-2
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_dec_2')(x)
    x = layers.LeakyReLU(name='lrelu_dec_2')(x)
    
    # Block-3
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_dec_3')(x)
    x = layers.LeakyReLU(name='lrelu_dec_3')(x)
    
    # Block-4
    recons_image = layers.Conv2DTranspose(3, 3, 1,padding='same', activation='sigmoid', name='conv_transpose_4')(x)
    
    return recons_image


# Loss Functions
def mse_loss(y_true, y_pred):
    r_loss = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=[1, 2, 3])
    return 1000 * r_loss

def kl_loss(mean, log_var):
    kl_loss =  -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
    return kl_loss

def vae_loss(y_true, y_pred, mean, log_var):
    r_loss = mse_loss(y_true, y_pred)
    latent_kl_loss = kl_loss(mean, log_var)
    return  r_loss + latent_kl_loss