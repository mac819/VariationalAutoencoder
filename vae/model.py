import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization


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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class EncoderLayer(keras.layers.Layer):
    def __init__(self, filters, kernel, stride, 
                trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        if name is not None:
            conv_name = name + "conv_layer"
        else:
            conv_name = None
        self.conv2d = layers.Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding="same", name=conv_name)
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU()
        self.pool = layers.MaxPool2D


    def call(self, inputs, *args, **kwargs):
        x = self.conv2d(inputs)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x

    # def add_config(self):
    #     return {'filters': self.filters, 'kernel': self.kernel, 'stride': self.stride}

class Encoder(keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = EncoderLayer(filters=32, kernel=3, stride=1, name="enc_block_1")
        self.block2 = EncoderLayer(filters=64, kernel=3, stride=2, name="enc_block_2")
        self.block3 = EncoderLayer(filters=64, kernel=3, stride=2, name="enc_block_3")
        self.block4 = EncoderLayer(filters=64, kernel=3, stride=1, name="anc_block_4")
        self.flatten = Flatten()
        self.dl1 = layers.Dense(131702, name="enc_dense_1")
        self.dl2 = layers.Dense(65536, name="enc_dense_2")
        self.dl3 = layers.Dense(32678, name="enc_dense_3")
        self.dl4 = layers.Dense(8192, name="enc_dense_4")
        self.dl5 = layers.Dense(1024, name="enc_dense_5")
        self.dl6 = layers.Dense(128, name="enc_dense_6")
        self.dl7 = layers.Dense(16, name="enc_dense_7")
        self.dense_layer_mean = Dense(2, name="mean")
        self.dense_layer_var = Dense(2, name="log_var")
        self.sampling = Sampling()

    def call(self, inputs, training=None, mask=None):
        
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        # x = self.dl1(x)
        # x = self.dl2(x)
        # x = self.dl3(x)
        # x = self.dl4(x)
        # x = self.dl5(x)
        # x = self.dl6(x)
        # x = self.dl7(x)
        mean = self.dense_layer_mean(x)
        log_var = self.dense_layer_var(x)

        z = self.sampling((mean, log_var))        
        return mean, log_var, z

    # def add_config(self):
    #     return {
    #             "enc_block1": self.block1,
    #             "enc_block2": self.block2,
    #             "enc_block3": self.block3,
    #             "enc_block4": self.block4,
    #             "enc_flatten": self.flatten,
    #             "enc_dense_mean": self.dense_layer_mean,
    #             "enc_dense_log_var": self.dense_layer_var,
    #         }

class DecoderLayer(keras.layers.Layer):

    def __init__(self, filters, kernel, stride, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        if name is not None:
            conv2dtr_name = name + "conv2d_trans"
        else:
            conv2dtr_name = None
        self.conv2dtrans = layers.Conv2DTranspose(
            filters=filters, 
            kernel_size=kernel, 
            strides=stride,
            padding="same", 
            name=conv2dtr_name)
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU()

    def call(self, inputs, *args, **kwargs):

        x = self.conv2dtrans (inputs)
        x = self.batchnorm(x)
        return self.activation(x)
    
    # def add_config(self):
    #     return {'filters': self.filters, 'kernel': self.kernel, 'stride': self.stride}

class Decoder(keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dl1 = layers.Dense(16, name="dec_dense_1")
        self.dl2 = layers.Dense(128, name="dec_dense_2")
        self.dl3 = layers.Dense(1024, name="dec_dense_3")
        self.dl4 = layers.Dense(8192, name="dec_dense_4")
        self.dl5 = layers.Dense(32678, name="dec_dense_5")
        self.dl6 = layers.Dense(65536, name="dec_dense_6")
        self.dl7 = layers.Dense(131702, name="dec_dense_7")
        self.dec_dense_layer = layers.Dense(262144, name="dec_dense")
        self.layer_reshape = layers.Reshape((64, 64, 64), name='Reshape_Layer')
        self.block1 = DecoderLayer(filters=64, kernel=3, stride=1)
        self.block2 = DecoderLayer(filters=64, kernel=3, stride=2)
        self.block3 = DecoderLayer(filters=32, kernel=3, stride=2)

        self.block4 = layers.Conv2DTranspose(
            filters=3, 
            kernel_size=3, 
            strides=1, 
            activation="sigmoid",
            padding="same",
            name="recons_image_layer"
            )

    def call(self, inputs, training=None, mask=None):

        x = self.dl1(inputs)
        # x = self.dl2(x)
        # x = self.dl3(x)
        # x = self.dl4(x)
        # x = self.dl5(x)
        # x = self.dl6(x)
        # x = self.dl7(x)
        x = self.dec_dense_layer(x)
        x = self.layer_reshape(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return self.block4(x)

    # def add_config(self):
    #     return {
    #         'dense': self.dec_dense_layer,
    #         'reshape': self.layer_reshape,
    #         'block1': self.block1,
    #         'block2': self.block2,
    #         'block3': self.block3,
    #         'block4': self.block4
    #     }


class VariationalAutoEncoder(keras.Model):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, training=None, mask=None):

        mean, log_var, z = self.encoder(inputs)
        recons_img = self.decoder(z)

        return mean, log_var, recons_img



