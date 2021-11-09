import tensorflow as tf
from matplotlib.image import imread
from tensorflow import keras
import pandas as pd
import numpy as np
import streamlit as st



st.title("CAT-DOG prediction")
st.write("Select image to predict eg:")
st.image("https://cdn.britannica.com/91/181391-050-1DA18304/cat-toes-paw-number-paws-tiger-tabby.jpg",width=200)
@st.cache(allow_output_mutation= True)

def prediction(image):
    model=keras.models.load_model("cat-dog_model.h5")
    image_array = np.array(image)
    image = tf.expand_dims(tf.image.resize(image, (200,200)), axis=0)
    pred = model.predict(image)
    if pred == 0:
        pred_out = "Cat"
    else:
        pred_out = "Dog"
    return pred_out

upload_img=st.file_uploader("Upload image of cat or dog", type ="jpg")
if upload_img is not None:
    image=imread(upload_img)
    st.write("")
    st.write("Classifying image ....")
    predict = prediction(image)
    st.image(upload_img)
    st.write("Predicted *image* contains :")
    st.write(predict)

