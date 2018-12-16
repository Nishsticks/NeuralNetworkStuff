import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6)  # train the model


TestImg = cv2.imread('Test5Crop.jpg', 0)


retval, procImage = cv2.threshold(TestImg, 100, 255, cv2.THRESH_BINARY_INV)


newArray = np.zeros(shape=(10000, 28, 28))

myTest = tf.convert_to_tensor(newArray)



newArray[0] = TestImg
predictions = model.predict([newArray])

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
cv2.imshow('number', x_test[0])
cv2.imshow('New Window', procImage)
plt.imshow(newArray[0])

plt.show()


# for i in range(0, len(x_test)):
#     cv2.imshow('number list', x_test[i])


