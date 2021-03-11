import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))
 
train_data = np.random.randint(5, size=(10000,32,32,3))
val_data = np.random.randint(5, size=(1000,32,32,3))
test_data = np.random.randint(5, size=(1000,32,32,3))

train_labels = np.random.randint(10, size=10000)
val_labels = np.random.randint(10, size=1000)
test_labels = np.random.randint(10, size=1000)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=3, validation_data=(val_data, val_labels))

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
