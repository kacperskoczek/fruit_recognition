import numpy as np
import cv2
from tensorflow.keras.models import load_model

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('vgg19_mydataset_240_320.h5')

image_gen = ImageDataGenerator()
train_image_gen = image_gen.flow_from_directory('my_fruits_dataset',
                                           target_size=(240,320),
                                            color_mode='rgb',
                                           batch_size=32,
                                           class_mode='categorical')

def predict_image(my_image):
    my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
    my_image = tf.image.resize_with_pad(my_image, 240, 320)
    my_image = np.array(my_image)
    my_image = my_image/255
    my_image = np.expand_dims(my_image, axis=0)
    result = model.predict_classes(my_image)

    class_names = train_image_gen.class_indices
    result_class = list(class_names.keys())[list(class_names.values()).index(result)]
    score = float("%0.2f" % (max(model.predict(my_image)[0]) * 100))

    return result_class, score

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)

    k = cv2.waitKey(10)
    if k == 32:
        print(predict_image(frame))

    if k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()