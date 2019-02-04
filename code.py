import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers.core import Dense,Flatten
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
%matplotlib inline

train_path = '../input/alphabets-sign-language/asl_alphabet_1/asl_alphabet_train/'

valid_path = '../input/alphabets-sign-language/asl_alphabet_1/asl_alphabet_valid/'
preprocess_func = keras.applications.mobilenet.preprocess_input
train_batches = ImageDataGenerator(preprocessing_function = preprocess_func).flow_from_directory(train_path,target_size=(224,224),batch_size=20)

valid_batches = ImageDataGenerator(preprocessing_function = preprocess_func).flow_from_directory(valid_path,target_size=(224,224),batch_size=20)
from keras.applications import MobileNet
model = MobileNet(weights='../input/mobilenet-h5/mobilenet_1_0_128_tf.h5')
model.summary()
type(model)

x = model.layers[-6].output
predictions = Dense(29,activation='softmax')(x)

model = Model(inputs = model.input, outputs=predictions)
model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False
    
model.compile(Adam(lr = 0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit_generator(train_batches, steps_per_epoch = 3627, validation_data = valid_batches, validation_steps = 723, epochs = 10, verbose = 2)

model.save('sign_language.h5')

####### model build and saved upto here

from keras.models import load_model

class SignLanguageModel():

    Signs = ["A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                     "del", "nothing",
                     "space"]
                     

    def __init__(self):
        # load model from JSON file
        self.model = load_model("sign_language.h5")


    def predict_sign(self, img):
        self.preds = self.model.predict(img)
        return FacialExpressionModel.Signs(self.preds)
        
import cv2
from model import SignLanguageModel


hand = cv2.CascadeClassifier('Hand.Cascade.1.xml')
model = FacialExpressionModel()
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_sign(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
