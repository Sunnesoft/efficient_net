from abc import ABC, ABCMeta
import math

import tensorflow as tf
import tensorflow_datasets as tfds

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
                 dropout_batch_rate=0.2,
                 image_resolution=224, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)

        self._divisor = 8

        self._width_coefficient = width_coefficient
        self._depth_coefficient = depth_coefficient
        self._se_ratio = se_ratio
        self._dropout_rate = dropout_rate
        self._dropout_batch_rate = dropout_batch_rate
        self._image_resolution = image_resolution

        self.list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        self.list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        self.expand_rates = [1, 6, 6, 6, 6, 6, 6]
        self.strides = [1, 2, 2, 2, 1, 2, 1]
        self.kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        self.list_channels = [EfficientNet.round_filters(
            c, self.width_coefficient, self._divisor) for c in self.list_channels]

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
                drop_rate = self.dropout_batch_rate * block_counter / num_blocks
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
        self.ll.append(Dropout(self.dropout_rate, name='EffNet_drop'))
        self.ll.append(Dense(num_classes, activation=tf.keras.activations.softmax, name='EffNet_fc'))

        self.nn = Sequential(self.ll)

    def call(self, inputs, training=None, mask=None):
        return self.nn(inputs)

    @property
    def width_coefficient(self):
        return self._width_coefficient

    @property
    def depth_coefficient(self):
        return self._depth_coefficient

    @property
    def se_ratio(self):
        return self._se_ratio

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def dropout_batch_rate(self):
        return self._dropout_batch_rate

    @property
    def image_resolution(self):
        return self._image_resolution

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
        super(EfficientNetB0, self).__init__(num_classes, 1, 1, se_ratio, 0.2, 0.2, 224, **kwargs)


class EfficientNetB1(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB1, self).__init__(num_classes, 1, 1.1, se_ratio, 0.2, 0.2, 240, **kwargs)
        self.resolution = 224


class EfficientNetB2(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB2, self).__init__(num_classes, 1.1, 1.2, se_ratio, 0.3, 0.3, 260, **kwargs)


class EfficientNetB3(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB3, self).__init__(num_classes, 1.2, 1.4, se_ratio, 0.3, 0.3, 300, **kwargs)


class EfficientNetB4(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB4, self).__init__(num_classes, 1.4, 1.8, se_ratio, 0.4, 0.4, 380, **kwargs)


class EfficientNetB5(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB5, self).__init__(num_classes, 1.6, 2.2, se_ratio, 0.4, 0.4, 456, **kwargs)


class EfficientNetB6(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB6, self).__init__(num_classes, 1.8, 2.6, se_ratio, 0.5, 0.5, 528, **kwargs)


class EfficientNetB7(EfficientNet, metaclass=ABCMeta):
    def __init__(self,
                 num_classes=100,
                 se_ratio=0.25, **kwargs):
        super(EfficientNetB7, self).__init__(num_classes, 2.0, 3.1, se_ratio, 0.5, 0.5, 600, **kwargs)


class DatasetPreprocessor:
    def __init__(self, classes_number, target_width, target_height, resize_method='bilinear'):
        self.classes_number = classes_number
        self.target_width = target_width
        self.target_height = target_height
        self.resize_method = resize_method

    def process_as_numpy(self, ds):
        """
        Ex.:
                [ds_train, ds_test], ds_info = tfds.load('cifar100', split=['train', 'test'],
                                             shuffle_files=True, with_info=True,
                                             as_supervised=True, batch_size=-1)

                dspr = DatasetPreprocessor(CLASSES_NUMBER, en.image_resolution, en.image_resolution)
                ds_train_cpu = dspr.process_as_numpy(tfds.as_numpy(ds_train))

        :param ds: dataset, which must loaded with keys as_supervised=True and batch_size=-1
        :return: list of objects
        """

        def run(x, y):
            with tf.device('/device:CPU:0'):
                x_new = tf.image.resize(x, [self.target_height, self.target_width], method=self.resize_method)
                y_new = [0] * self.classes_number
                y_new[y] = 1
                return x_new, y_new

        return list(map(run, ds[0], ds[1]))

    def process_as_dataset(self, ds):
        return ds.map(
            lambda x, y: (tf.image.resize(
                x, [self.target_height, self.target_width], method=self.resize_method) / 255.,
                          tf.one_hot(y, self.classes_number, dtype=tf.uint32)))


if __name__ == '__main__':
    print(tf.__version__)
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
    print("Num CPUs Available: ", tf.config.list_physical_devices('CPU'))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.run_functions_eagerly(True)

    BATCH_SIZE = 8
    CHANNEL_COUNT = 3
    CLASSES_NUMBER = 100

    with tf.device('/device:GPU:0'):
        en = EfficientNetB0(CLASSES_NUMBER)
        en.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                   loss=tf.keras.losses.CategoricalCrossentropy(),
                   metrics=['accuracy'])
        en.build((BATCH_SIZE, en.image_resolution, en.image_resolution, CHANNEL_COUNT))
        print(en.summary())

        [ds_train, ds_test], ds_info = tfds.load('cifar100', split=['train', 'test'],
                                                 shuffle_files=True, with_info=True,
                                                 as_supervised=True)

        dspr = DatasetPreprocessor(CLASSES_NUMBER, en.image_resolution, en.image_resolution)
        ds_train_gpu = dspr.process_as_dataset(ds_test)
        print(ds_train_gpu)
        # ds_train_cpu = dspr.process_as_numpy(tfds.as_numpy(ds_train))

        ds_train_gpu = ds_train_gpu.batch(BATCH_SIZE).prefetch(1)
        print(ds_train_gpu)

        en.fit(ds_train_gpu, epochs=5)

    exit(0)
