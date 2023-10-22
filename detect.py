import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

gender_model = load_model('models\gender.model')
hijab_model = load_model('models\hijab.model')
gender_classes = ['man','woman']
hijab_classes = ['hijab','no hijab']

def detect_gender(image):
    img = cv2.imread(image)
    dim = (96, 96)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("face.jpg", resized)
    img = cv2.imread('face.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap=plt.cm.binary)
    prediction = gender_model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    prediction = gender_classes[index]
    print(f'prediction is {gender_classes[index]}')
    os.remove('face.jpg')
    return prediction

def detect_hijab(image):
    img = cv2.imread(image)
    dim = (224, 224)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("face.jpg", resized)
    img = cv2.imread('face.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap=plt.cm.binary)
    prediction = hijab_model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    prediction = hijab_classes[index]
    print(f'prediction is {hijab_classes[index]}')
    os.remove('face.jpg')
    return prediction