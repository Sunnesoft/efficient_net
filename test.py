import tensorflow as tf


with tf.device('/device:CPU:0'):
    BATCH_SIZE = 8
    CHANNEL_COUNT = 3
    CLASSES_NUMBER = 10
    en = tf.keras.applications.EfficientNetB0()
    en.compile(optimizer=tf.keras.optimizers.RMSprop(),
               loss=tf.keras.losses.CategoricalCrossentropy(),
               metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy()])
    en.build((BATCH_SIZE, 224, 224, CHANNEL_COUNT))
    print(en.summary())
