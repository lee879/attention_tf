import tensorflow as tf
from tensorflow.python.keras import Model,Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Activation,Layer,Dropout,MaxPooling2D,GlobalMaxPooling2D,UpSampling2D
import cv2
import numpy as np

class channel_attention(Layer):
    def __init__(self,c):
        super(channel_attention, self).__init__()
        self.mlp = Sequential([
            Dense(c,activation=tf.nn.relu),
            MaxPooling2D(2),
            Dense(c / 2,activation=tf.nn.relu),
            UpSampling2D(2),
            Dense(c,activation=tf.nn.relu)
        ])
        self.sigm = Activation(tf.nn.sigmoid)
    def call(self, inputs, **kwargs):
        x = self.mlp(inputs)
        x1 = self.sigm(x)

        return x1*inputs

class Spatial_attention(Layer):
    def __init__(self,c,r):
        super(Spatial_attention, self).__init__()
        self.conv_1 = Conv2D(c/r,7,1,"same",activation=tf.nn.relu)
        self.conv_2 = Conv2D(c,7,1,"same",activation=tf.nn.relu)
        self.sigm = Activation(tf.nn.sigmoid)
    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x1 = self.conv_2(x)
        x2 = self.sigm(x1)
        return x2*inputs

class Global_Attention(Layer):
    def __init__(self,channels,radio):
        super(Global_Attention, self).__init__()
        self.ca = channel_attention(c=channels)
        self.sa = Spatial_attention(c=channels,r=radio)
    def call(self, inputs, **kwargs):
        x = self.ca(inputs)
        out = self.sa(x)
        return out
