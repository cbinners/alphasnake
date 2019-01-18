import tensorflow as tf

tf.enable_eager_execution()


class Net():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(21, 21, 3)),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.train.AdamOptimizer(),
                           metrics=['accuracy'])

        print(self.model)

    def update(self, state, score):
        # TODO: Update the net with the state and score
        input = tf.random.uniform((1, 21, 21, 3))
        pass

    def predict(self):
        # TODO: Get the score of state
        data = tf.random.uniform((1, 21, 21, 3))
        result = self.model.predict(data, batch_size=1).mean()
        return result
