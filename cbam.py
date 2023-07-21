import tensorflow as tf
from tensorflow.python.keras.layers import AveragePooling2D,MaxPooling2D,Dense,Conv2D,Activation,Layer,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.python.keras import Model ,Sequential

class CAM(Layer):
    def __init__(self,c,r):
        super(CAM, self).__init__()
        self.gmp = GlobalMaxPooling2D()
        self.gap = GlobalAveragePooling2D()
        self.mlp = Sequential([
            Dense(c/r,activation=tf.nn.relu),
            Dense(c,activation=tf.nn.relu)
        ])
        self.ac = Activation(tf.nn.sigmoid)

    def call(self, inputs, *args, **kwargs):
        x = tf.expand_dims(tf.expand_dims(self.gap(inputs),axis=1),axis=1)
        y = tf.expand_dims(tf.expand_dims(self.gmp(inputs),axis=1),axis=1)
        x1 = self.mlp(x)
        y1 = self.mlp(y)
        z = self.ac(tf.add(x1,y1))
        out = inputs * tf.tile(z,(1,inputs.shape[1],inputs.shape[1],1))
        return out

class SAM(Layer):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = Conv2D(1,7,1,"same",activation=tf.nn.sigmoid)
    def call(self, inputs, *args, **kwargs):
        x = tf.math.reduce_max(inputs,axis=-1,keepdims=True)
        y = tf.math.reduce_mean(inputs,axis=-1,keepdims=True)
        z = tf.concat([x,y],axis=-1)
        out = tf.tile(self.conv(z),(1,1,1,inputs.shape[-1])) * inputs
        return out

class CBAM(Model):
    def __init__(self,channels,radio):
        super(CBAM, self).__init__()
        self.cam = CAM(c=channels,r=radio)
        self.sam = SAM()
    def call(self, inputs, training=None, mask=None):
        x = self.cam(inputs)
        out = self.sam(x)
        return out
    
# m = CAM(c=16,r=2)
m = CBAM(channels=16,radio=2)
x = tf.random.normal(shape=(8,512,512,16))
y = m(x)

print(y.shape)