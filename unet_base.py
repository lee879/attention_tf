import tensorflow as tf
from burpool_tf import burpool
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Conv2DTranspose,concatenate,Activation,UpSampling2D,Flatten,Dense
from tensorflow.python.keras import Model

class block_1(tf.keras.layers.Layer):
    def __init__(self,filt):
        super(block_1, self).__init__()
        self.block = tf.keras.Sequential([
            Conv2D(filt,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(filt,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
        ])
    def call(self, inputs, *args, **kwargs):
        return self.block(inputs)

class block_2(tf.keras.layers.Layer):
    def __init__(self,filt):
        super(block_2, self).__init__()
        self.block = tf.keras.Sequential([
            Conv2D(filt,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(filt / 2,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
        ])
    def call(self, inputs, *args, **kwargs):
        return self.block(inputs)

class block_3(tf.keras.layers.Layer):
    def __init__(self,filt):
        super(block_3, self).__init__()
        self.block = tf.keras.Sequential([
            Conv2D(filt,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(filt / 2,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
        ])
    def call(self, inputs, *args, **kwargs):
        return self.block(inputs)


class block_4(tf.keras.layers.Layer):
    def __init__(self,filt):
        super(block_4, self).__init__()
        self.block = tf.keras.Sequential([
            Conv2D(filt,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(filt,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
        ])

    def call(self, inputs, *args, **kwargs):
        return self.block(inputs)

class UNet(Model):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.conv1 = block_1(64)
        self.down1 = burpool(64)

        self.conv2 = block_1(128)
        self.down2 = burpool(128)

        self.conv3 = block_1(256)
        self.down3 = burpool(256)

        self.conv4 = block_1(512)
        self.down4 = burpool(512)

        self.conv5 = block_2(1024)

        self.up = UpSampling2D(2)
        self.conv6 = block_3(512)
        self.conv7 = block_3(256)
        self.conv8 = block_3(128)
        self.conv9 = block_4(64)
        self.out = tf.keras.Sequential([
            Conv2D(num_classes,1,1,"same",activation=tf.nn.softmax)
        ])

    def call(self, inputs, training=None, mask=None):
        # this is encodering
        x1 = self.conv1(inputs)
        x = self.down1(x1)

        x2 = self.conv2(x)
        x = self.down2(x2)

        x3 = self.conv3(x)
        x = self.down3(x3)

        x4 = self.conv4(x)
        x = self.down4(x4)

        x5 = self.conv5(x)

        # this is decodering
        y1 = self.up(x5)
        y = tf.concat([y1,x4],axis=-1)
        y2 = self.conv6(y)

        y3 = self.up(y2)
        y = tf.concat([y3,x3],axis=-1)
        y4 = self.conv7(y)

        y5 = self.up(y4)
        y = tf.concat([y5,x2],axis=-1)
        y6 = self.conv8(y)

        y7 = self.up(y6)
        y = tf.concat([y7,x1],axis=-1)
        y8 = self.conv9(y)

        return self.out(y8)

# model = UNet(num_classes=4)
# x = tf.random.normal(shape=[1,512,512,3])
# y = model(x)
# model.summary()
# print(y.shape)