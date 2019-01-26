import tensorflow as tf
import numpy as np
import os
import time


class Net():
    def __init__(self, path=None):
        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)
        self.checkpoint_path = path

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=False, save_weights_only=True)

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(10, 10, 3), name="input"),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3),
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, 'tanh', name="infer")
        ])

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=0.002)

        self.model.compile(loss=tf.keras.losses.mean_squared_error,
                           optimizer=tf.train.AdamOptimizer(), metrics=['mae'])

        if tf.train.checkpoint_exists(self.checkpoint_path):
            print("Loading checkpoint")
            self.model.load_weights(self.checkpoint_path)
        else:
            print("No existing checkpoint, create new")

    def update(self, x, y):
        tf_X = tf.convert_to_tensor(x, dtype=tf.float32)
        tf_Y = tf.convert_to_tensor(y, dtype=tf.float32)
        self.model.fit(tf_X, tf_Y, callbacks=[
                       self.cp_callback], epochs=3, steps_per_epoch=10)

    def predict(self, state):
        tensor = tf.convert_to_tensor(np.asarray(state))
        result = self.model.predict(tensor, steps=1)
        return result

    def reload(self):
        self.model.load_weights(self.checkpoint_path)

    def save(self):
        # Use TF to save the graph model instead of Keras save model to load it in Golang
        builder = tf.saved_model.builder.SavedModelBuilder("models/10x10-save")
        # Tag the model, required for Go
        builder.add_meta_graph_and_variables(self.sess, ["mytag"])
        builder.save()
