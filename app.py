import streamlit as st
from tensorflow import keras
import tensorflow as tf
import cv2
st.write("This is a simple image classification web app to predict if a given X-ray contains Pneumonia or not")
file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
from PIL import Image, ImageOps
import numpy as np
with open('model_2.json', 'r') as json_file:
    json_savedModel= json_file.read()
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('model_p.h5')

def teachable_machine_classification(img, weights_file):


    

   
    data = np.ndarray(shape=(1, 224, 224,3), dtype=np.float32)
    image = img
   
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

   
    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 255) 

    
    data[0] = normalized_image_array
    
    # img = np.array(img.resize(224,224),dtype=np.float32).reshape(1,224,224,3)
    # print(img.shape)
    # img = img/255
    # image = cv2.resize(img, (224,224))
    # image = image.astype('float32') / 255
    # image = np.expand_dims(image, axis=0)




    
    prediction = model.predict(data)
    
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    # image = cv2.imread(file)
    img = Image.open(file).convert('RGB')
    

    
    st.image(img, use_column_width=True)
    prediction = teachable_machine_classification(img, model)
    pred = float(prediction[0])
    
    
    if pred<0.9:
        st.write("It is Normal!")
    elif pred>=0.9:
        st.write("It is a Pneumonia!")
        st.text("Probability of Prediction is:")
        st.write(pred)
         
    
    
   
