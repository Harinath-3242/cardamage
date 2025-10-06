import streamlit as st
import numpy as np
import keras
import numpy as np
import keras
import keras.utils as im
import matplotlib.pyplot as plt
from keras.models import Model
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import pickle




def predict(model, image):
    
    image = image.convert('RGB')  
    image = image.resize((224, 224))
    x = im.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)


    output = np.argmax(model.predict(img_data), axis=1)

    
    index = ['Minor', 'Moderate', 'Severe', 'Good']
    result = index[output[0]]

    return result




def main():
    st.title("Car Damage Severity Assessmnent System")

    
    page = st.sidebar.selectbox("Select a page", ["Model Prediction", "About the Project"])

    if page == "Model Prediction":
        model_prediction_page()
    elif page == "About the Project":
        about_model_page()


def model_prediction_page():
    st.header("Model Prediction")

    
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
       


        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

       
        
        model = load_model("MobileNet_Model_Final.keras")
   
        with st.spinner("Making prediction..."):
            prediction = predict(model, image)

        st.subheader("Prediction Results")
        st.write(prediction)


    st.markdown("---")
    

def about_model_page():


    st.subheader("Overview")
    st.write("""
    This project aims to leverage Deep Learning techniques to assess the severity of car damage from images. The goal is to create an automated system that can quickly and accurately evaluate the extent of damage to a vehicle, assisting insurance companies, repair shops, and car owners in making informed decisions..
    """)

    st.subheader("Objectives")
    st.write("""
    1. **Automated Detection**: Identify and classify car damage from images.
    2. **Severity Assessment**: Categorize damage into levels (minor, moderate, severe).
    3. **Efficiency**: Ensure high accuracy and fast processing.
    """)

    st.subheader("Methodology")
    st.write("""
    - **Data Collection**: Images labeled with damage type and severity.
    - **Preprocessing**: Resize, normalize, and augment images.
    - **Model Development**: Use Convolutional Neural Networks (CNNs).
    - **Training & Validation**: Train model on split dataset and evaluate performance.
    - **Evaluation**: Use accuracy, precision, recall, and F1-score metrics.
    """)

    st.subheader("Technologies Used")
    st.write("""
    - **Framework**: TensorFlow/Keras or PyTorch
    - **Language**: Python
    - **Data Processing**: OpenCV, PIL
    - **Evaluation**: Scikit-learn
    """)



    st.subheader("Future Work")
    st.write("""
    - **Expand Dataset**: Include more diverse images.
    - **Real-world Application**: Integrate with industry applications.
    - **Enhance Model**: Use advanced techniques for better performance.
    """)

    st.subheader("Conclusion")
    st.write("""
    This project demonstrates the potential of deep learning to automate and improve car damage assessments, saving time and resources while ensuring reliable results.
    """)
    
    st.markdown("---")
    st.write("Developed by G Jayanth")

if __name__ == "__main__":
    main()
