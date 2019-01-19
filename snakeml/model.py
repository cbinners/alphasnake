import tensorflow as tf
import numpy as np
import os
import time

tf.enable_eager_execution()


class Net():
    def __init__(self, path):
        self.checkpoint_path = path
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                              save_weights_only=True,
                                                              verbose=1)

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(19, 19, 3)),
            tf.keras.layers.Conv2D(2, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.train.AdamOptimizer(
                               learning_rate=0.000001),
                           metrics=['accuracy'])

        self.reload()

    def update(self, state, score):
        tensorinput = [0, 0]
        tensorinput[score] = 1

        # Fill in the ys of shape (X, 2) where X is the number of inputs
        y = np.full((int(state.shape[0]), 2), tensorinput)

        # Convert np -> tf
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        # Fit
        self.model.fit(state, y, batch_size=32, callbacks=[self.cp_callback])

    def predict(self, state):
        now = time.time()
        result = self.model.predict(state)
        after = time.time()
        print(state.shape[0], "inputs in", int(
            round(1000*(after - now))), "ms")
        return result.mean(axis=0)[1]

    def reload(self):
        if os.path.isfile(self.checkpoint_path):
            self.model.load_weights(self.checkpoint_path)
