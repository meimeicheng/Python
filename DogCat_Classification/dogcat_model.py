import tensorflow as tf
from cv2 import cv2
import os

CATEGORIES = ["Cat", "Dog"]

#找回訓練架構
model= tf.keras.models.load_model('imageClassification.h5')
model.summary()
def prepare(filepath):
    IMG_SIZE = 150  # 150 in txt-based
    img_array = cv2.imread(filepath)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE , 3) # return the image with shaping that TF wants.
#放入使用資料   
prediction = model.predict([prepare('net-dog-1.jpg')]) 
print('Result  :' , CATEGORIES[int(prediction[0][0])])
# print('===prediction====' ,prediction)
           