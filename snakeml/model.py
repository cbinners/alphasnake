import tensorflow as tf
import numpy as np
import os
import time

tf.enable_eager_execution()


class Net():
    def __init__(self, path=None):
        self.checkpoint_path = path
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                              save_weights_only=True,
                                                              verbose=1)

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(19, 19, 3)),
            tf.keras.layers.Conv2D(2, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, 'tanh')
        ])

        self.model.compile(loss=tf.keras.losses.mean_squared_error,
                           optimizer=tf.train.AdamOptimizer(
                               learning_rate=0.0000001),
                           metrics=['mae'])

        self.reload()

    def update(self, state, score):
        y = np.full((int(state.shape[0]), 1), score)

        # Convert np -> tf
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        # Fit
        self.model.fit(state, y, batch_size=32, callbacks=[
                       self.cp_callback], epochs=1)

    def predict(self, state):
        result = self.model.predict(state)
        return result.mean()

    def reload(self):
        if self.checkpoint_path is not None:
            self.model.load_weights(self.checkpoint_path)
