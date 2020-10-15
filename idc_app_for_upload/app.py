# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

# Image
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2 as cv
import glob

# CNN
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# DB
# from managed_db import *
import sqlite3
conn = sqlite3.connect('usersdata.db')
c = conn.cursor()

# Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

# Process Image to Array
def preprocessed_image(file):
    image = file.resize((50,50), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    return image

# Plot Random Image
def show_image(path):
    fig = plt.figure(figsize= (10,10))
    ind = np.random.randint(0, len(path), 25)
    i=0
    for loc in ind:
        plt.subplot(5,5,i+1)
        sample = load_img(path[loc], target_size=(50,50))
        sample = img_to_array(sample)
        plt.axis("off")
        plt.imshow(sample.astype("uint8"))
        plt.title((os.path.split(path[loc])[1])[10:-11])
        i+=1
    return fig

# Feature Maps
def feature_of(feature_map, n_square):
    square = n_square
    ix = 1
    fig = plt.figure(figsize= (5, 5))
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_map[0, :, :, ix-1], cmap='gray')
            ix += 1
    return fig


def main():
    """Invasive Ductal Carcinoma Detection Using CNN"""
    st.title("Invasive Ductal Carcinoma Detection Using CNN")


    menu = ["Home", "Login", "Signup"]
    submenu = ["Plot", "Visualisasi IDC","Feature Maps","Prediction"]

    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("What is Invasive Ductal Carcinoma (IDC)?")
        st.markdown("#### Context")
        """
        Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign 
        an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions 
        which contain the IDC. As a result, one of the common pre-processing steps for automatic 
        aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide.
        """
        st.markdown("#### Content")
        """
        The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens 
        scanned at 40x. From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative 
        and 78,786 IDC positive). Each patch’s file name is of the format: uxXyYclassC.png — > example 
        10253idx5x1351y1101class0.png . Where u is the patient ID (10253idx5), X is the x-coordinate of 
        where this patch was cropped from, Y is the y-coordinate of where this patch was cropped from, 
        and C indicates the class where 0 is non-IDC and 1 is IDC.
        """
        st.markdown("#### Acknowledgements")
        """
        The original files are located here: http://gleason.case.edu/webdata/jpi-dl-tutorial/IDC_regular_ps50_idx5.zip
        Citation: https://www.ncbi.nlm.nih.gov/pubmed/27563488 and http://spie.org/Publications/Proceedings/Paper/10.1117/12.2043872
        """
        st.markdown("#### Inspiration")
        """
        Breast cancer is the most common form of cancer in women, and invasive ductal carcinoma (IDC) is 
        the most common form of breast cancer. Accurately identifying and categorizing breast cancer 
        subtypes is an important clinical task, and automated methods can be used to save time and reduce error.
        """
    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pwsd = generate_hashes(password)
            result = login_user(username, verify_hashes(password, hashed_pwsd))
            if result:
                st.success("Welcome {}".format(username))

                activity = st.selectbox("Activity", submenu)
                if activity == "Plot":
                    st.subheader("Data Plot")

                    status = st.radio("Data Distribution", ("Data raw", "Data preprocessed"))

                    if status == 'Data raw':
                        img = Image.open(os.path.join("data/sns.countplot(y_train).jpeg"))
                        st.image(img, width=300, caption="Data Train")

                        img = Image.open(os.path.join("data/sns.countplot(y_test).jpeg"))
                        st.image(img, width=300, caption="Data Test")
                    else:
                        img = Image.open(os.path.join("data/sns.countplot(y_train2).jpeg"))
                        st.image(img, width=300, caption="Data Train")

                        img = Image.open(os.path.join("data/sns.countplot(y_test2).jpeg"))
                        st.image(img, width=300, caption="Data Test")
                
                elif activity == "Visualisasi IDC":
                    st.subheader("Visualisasi IDC(-/+)")
                    sample_gambar = st.radio("Few example of IDC with its coordinate", ("IDC (-)", "IDC (+)"))
                    if sample_gambar == 'IDC (-)':
                        figure_path = glob.glob("gambar visual/0/*.png", recursive=True)
                        figure = show_image(figure_path)
                        st.pyplot(figure)
                    else:
                        figure_path = glob.glob("gambar visual/1/*.png", recursive=True)
                        figure = show_image(figure_path)
                        st.pyplot(figure)
                
                elif activity == "Feature Maps":
                    st.subheader("Feature Maps")
                    feature_maps = st.radio("Visualization Feature Maps from hidden layer", ("VGG16", "5 Layers Conv2d"))
                    if feature_maps == 'VGG16':
                        model_ = load_model(os.path.join("models/vgg-model-weights-improvement-the-best.h5"))
                        model_baru = model_.layers[0] # Khusus vgg
                        model_baru = Model(inputs=model_baru.inputs, outputs=model_baru.layers[1].output)
                        model_baru.summary()

                        img = Image.open(os.path.join("gambar visual/0/9178_idx5_x2651_y1251_class0.png"))
                        img = preprocessed_image(img)
                        img = preprocess_input(img)
                        feature_maps = model_baru.predict(img)
                       
                        figure = feature_of(feature_maps, 8)
                        st.pyplot(figure)
                    else:
                        model_ = load_model(os.path.join("models/weights-improvement-the-best.h5"))
                        model_baru = model_
                        model_baru = Model(inputs=model_baru.inputs, outputs=model_baru.layers[1].output)
                        model_baru.summary()

                        img = Image.open(os.path.join("gambar visual/0/9178_idx5_x2651_y1251_class0.png"))
                        img = preprocessed_image(img)
                        img = preprocess_input(img)
                        feature_maps = model_baru.predict(img)
                       
                        figure = feature_of(feature_maps, 5)
                        st.pyplot(figure)

                elif activity == "Prediction":
                    st.subheader("Predictive Analytics")

                    # Upload Image
                    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

                    if image_file is not None:
                        our_image = Image.open(image_file)
                        st.text("Image Uploaded!")
                        st.image(our_image)

                        # Processed Image
                        image_test = preprocessed_image(our_image)
                    else:
                        st.warning("Please upload the image!")
                    
                    # ML / Predict Image
                    model_choice = st.selectbox("Select Model", ["VGG16", "5 Layers Conv2d"])
                    if st.button("Predict"):
                        if model_choice == "VGG16":
                            model_ = load_model(os.path.join("models/vgg-model-weights-improvement-the-best.h5"))
                            opt = SGD(lr=0.001, momentum = 0.9)
                            model_.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                            prediction = model_.predict(image_test)
                            prediction_result = np.argmax(prediction[0])
                            
                        elif model_choice == "5 Layers Conv2d":
                            model_ = load_model(os.path.join("models/weights-improvement-the-best.h5"))
                            opt = SGD(lr=0.001, momentum = 0.9)
                            model_.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                            prediction = model_.predict(image_test)
                            prediction_result = np.argmax(prediction[0])

                        # st.write(prediction_result)
                        if prediction_result == 1 :
                            st.warning("Patient's positive IDC!")
                            st.error("Please seek for treatment and keep healthy lifestyle!")
                        else:
                            st.success("It's negative!")
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "Signup":
        new_username = st.text_input("user name")
        new_password = st.text_input("Password", type='password')

        confirm_password = st.text_input("Confirm Password", type='password')
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")
        
        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username, hashed_new_password)
            st.success("You have successfully created a new account")
            st.info("Login to Get Started ")

if __name__ == '__main__':
    main()