import tensorflow as tf
import numpy as np
import os
import time


class Net():
    def __init__(self, path=None):
        self.checkpoint_path = path
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                              save_weights_only=True,
                                                              verbose=1)

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(23, 23, 3)),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, 'tanh')
        ])

        self.model.compile(loss=tf.keras.losses.mean_squared_error,
                           optimizer=tf.train.AdamOptimizer(
                               learning_rate=0.0000001),
                           metrics=['mae'])

        # self.reload()

    def update(self, x, y):
        tf_X = tf.convert_to_tensor(x, dtype=tf.float32)
        tf_Y = tf.convert_to_tensor(y, dtype=tf.float32)
        self.model.fit(tf_X, tf_Y, callbacks=[
                       self.cp_callback], epochs=10)

    def predict(self, state):
        tensor = tf.convert_to_tensor(state)
        result = self.model.predict(tensor)
        return result

    def reload(self):
        if self.checkpoint_path is not None:
            print("Loaded model weights...")
            self.model.load_weights(self.checkpoint_path)
