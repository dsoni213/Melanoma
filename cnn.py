from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2
import matplotlib.pyplot as plt

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (200, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (200, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

hist = classifier.fit_generator(training_set,
                         steps_per_epoch = 10 ,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 50)
classifier.save('C:/Users/vicky/Downloads/python-projects-master/python-projects-master/skin caner_new/model.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('melanoma.jpg', target_size = (200, 200))
test_image1 = image.img_to_array(test_image)
test_image1 = np.expand_dims(test_image1, axis = 0)
result = classifier.predict(test_image1)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'Melanoma'
	print(prediction)
else:
	prediction = 'Nonmelonama'
	print(prediction)

plt.figure()
plt.title("Accuracy")
plt.plot(hist.history['accuracy'], 'r', label='Training')
plt.plot(hist.history['val_accuracy'], 'b', label='Testing')
plt.legend()
plt.show()

plt.figure()
plt.title("Loss")
plt.plot(hist.history['loss'], 'r', label='Training')
plt.plot(hist.history['val_loss'], 'b', label='Testing')
plt.legend()
plt.show()
