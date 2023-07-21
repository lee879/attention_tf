import tensorflow as tf
from tensorflow.python.keras import Model,Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Activation,Layer,Dropout,MaxPooling2D,GlobalMaxPooling2D
import cv2
import numpy as np

class se(Layer):
    def __init__(self,c):
        super(se, self).__init__()
        self.pool = GlobalMaxPooling2D()
        self.mlp = Sequential([
         Dense(c,activation=tf.nn.relu),
         Dense(c,activation=tf.nn.relu)
        ]
        )
        self.scale_1 = tf.keras.layers.LayerNormalization()
    def call(self, inputs, **kwargs):
        x = tf.expand_dims(tf.expand_dims(self.pool(inputs),axis=1),axis=1)
        x1 = self.mlp(x)
        out = inputs * tf.tile(x1,(1,inputs.shape[1],inputs.shape[1],1))
        return out
