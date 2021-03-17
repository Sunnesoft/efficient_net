from abc import ABC, ABCMeta
import math

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, \
    DepthwiseConv2D, GlobalAvgPool2D, Multiply, Layer, Reshape, Dropout, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras import backend as K


class SqueezeExcitation(Layer):
    def __init__(self, filters, se_filters, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)

        self.filters = filters
        self.squeezed_filters = se_filters

        self.avg = GlobalAvgPool2D(name='%s_avg' % self.name)
        self.reshape = Reshape((1, 1, filters), name='%s_rshp' % self.name)

        self.fc1 = Conv2D(self.squeezed_filters, 1,
                          strides=(1, 1),
                          padding='same',
                          activation='swish',
                          name='%s_fc1' % self.name)

        self.fc2 = Conv2D(filters, 1,
                          strides=(1, 1),
                          padding='same',
                          activation='sigmoid',
                          name='%s_fc2' % self.name)
        self.mult = Multiply(name='%s_mult' % self.name)

    def call(self, inputs):
        x = self.avg(inputs)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.mult([inputs, x])


class MBConv(Layer):
    """
        MBConv+IdentitySkip: -X-> Conv|BN|Swish -> DepthwiseConv|BN|Swish -> SqueezeExcitation -> Conv|BN-(+X)->
    """

    def __init__(self,
                 filters,
                 filters_out,
                 expand_ratio,
                 se_ratio,
                 dw_strides,
                 kernel_size,
                 dropout_rate=None, **kwargs):
        super(MBConv, self).__init__(**kwargs)

        self.identity_skip = dw_strides == 1

        self.kernel_size = kernel_size
        self.filters = filters
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.expand_filters = int(self.filters * self.expand_ratio)
        self.squeezed_filters = max(1, int(self.filters * self.se_ratio))
        self.dropout_rate = dropout_rate

        self.expand = Conv2D(self.expand_filters, 1,
                             strides=(1, 1),
                             padding='same',
                             use_bias=False,
                             name='%s_expand' % self.name)
        self.bn1 = BatchNormalization(name='%s_bn1' % self.name)
        self.swish1 = Activation(tf.nn.swish, name='%s_swish1' % self.name)

        self.dwconv = DepthwiseConv2D(self.kernel_size, padding='same',
                                      strides=dw_strides,
                                      use_bias=False, name='%s_dwconv' % self.name)
        self.bn2 = BatchNormalization(name='%s_bn2' % self.name)
        self.swish2 = Activation(tf.nn.swish, name='%s_swish2' % self.name)

        self.se = SqueezeExcitation(self.expand_filters,
                                    self.squeezed_filters,
                                    name=self.name + '_SE')

        self.conv = Conv2D(filters_out, 1,
                           strides=(1, 1),
                           padding='same',
                           use_bias=False,
                           name='%s_conv' % self.name)
        self.bn3 = BatchNormalization(name='%s_bn3' % self.name)

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = self.bn1(x)
        x = self.swish1(x)

        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.swish2(x)

        x = self.se(x)

        x = self.conv(x)
        x = self.bn3(x)

        if inputs.shape == x.shape and self.identity_skip:
            x = self._dropout(x, training)
            x += inputs

        return x

    def _dropout(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        if (not training) or (self.dropout_rate is None):
            return inputs

        keep_prob = 1.0 - self.dropout_rate
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.divide(inputs, keep_prob) * binary_tensor
        return output


class EfficientNet(Model, ABC):
    def __init__(self,
                 num_classes=100,
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 se_ratio=0.25,
                 dropout_rate=0.2,
                 dropout_batch_rate=0.2, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8
        self.se_ratio = se_ratio

        self.list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        self.list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        self.expand_rates = [1, 6, 6, 6, 6, 6, 6]
        self.strides = [1, 2, 2, 2, 1, 2, 1]
        self.kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        self.list_channels = [EfficientNet.round_filters(
            c, self.width_coefficient, self.divisor) for c in self.list_channels]

        self.list_num_repeats = [EfficientNet.round_repeats(
            r, self.depth_coefficient) for r in self.list_num_repeats]

        self.ll = [
            Conv2D(self.list_channels[0], 2,
                   strides=2,
                   padding='same',
                   use_bias=False,
                   name='EffNet_conv1'),
            BatchNormalization(name='EffNet_bn1'),
            Activation(tf.nn.swish, name='EffNet_swish1')]

        block_counter = 0
        num_blocks = sum(self.list_num_repeats)
        for i in range(7):
            ch = self.list_channels[i + 0]
            ch_next = self.list_channels[i + 1]

            for j in range(self.list_num_repeats[i]):
                stride = self.strides[i] if j == 0 else 1
                chN = ch if j == 0 else ch_next
                drop_rate = dropout_batch_rate * block_counter / num_blocks
                self.ll.append(MBConv(
                    chN,
                    ch_next,
                    self.expand_rates[i],
                    self.se_ratio,
                    stride,
                    self.kernel_sizes[i],
                    drop_rate,
                    name='EffNet_MBConv%d_%d_%d' % (self.expand_rates[i], i, j)))
                block_counter += 1

        self.ll.append(Conv2D(self.list_channels[-1], 1, use_bias=False, name='EffNet_conv2'))
        self.ll.append(BatchNormalization(name='EffNet_bn2'))
        self.ll.append(Activation(tf.nn.swish, name='EffNet_swish2'))
        self.ll.append(GlobalAvgPool2D(name='EffNet_gloavg'))
        self.ll.append(Dropout(dropout_rate, name='EffNet_drop'))
        self.ll.append(Dense(num_classes, activation=tf.keras.activations.softmax, name='EffNet_fc'))

        self.nn = Sequential(self.ll)

    def call(self, inputs, training=None, mask=None):
        return self.nn(inputs)

    def round_filters(filters, multiplier, divisor):
        if not multiplier:
            return filters

        filters *= multiplier
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats, multiplier):
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))


class EfficientNetB0(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB0, self).__init__(num_classes, 1, 1, se_ratio, 0.2, 0.2, **kwargs)


class EfficientNetB1(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB1, self).__init__(num_classes, 1, 1.1, se_ratio, 0.2, 0.2, **kwargs)


class EfficientNetB2(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB2, self).__init__(num_classes, 1.1, 1.2, se_ratio, 0.3, 0.3, **kwargs)


class EfficientNetB3(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB3, self).__init__(num_classes, 1.2, 1.4, se_ratio, 0.3, 0.3, **kwargs)


class EfficientNetB4(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB4, self).__init__(num_classes, 1.4, 1.8, se_ratio, 0.4, 0.4, **kwargs)


class EfficientNetB5(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB5, self).__init__(num_classes, 1.6, 2.2, se_ratio, 0.4, 0.4, **kwargs)


class EfficientNetB6(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB6, self).__init__(num_classes, 1.8, 2.6, se_ratio, 0.5, 0.5, **kwargs)


class EfficientNetB7(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB7, self).__init__(num_classes, 2.0, 3.1, se_ratio, 0.5, 0.5, **kwargs)


if __name__ == '__main__':
    print(tf.__version__)
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
    print("Num CPUs Available: ", tf.config.list_physical_devices('CPU'))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    en = EfficientNetB7()
    en.compile(optimizer="adam", loss="mse")
    en.build((1024, 256, 256, 3))
    print(en.summary())

    exit(0)
