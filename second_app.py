import streamlit as st

# Text/Title
st.title("Streamlit Tutorials")

# Header/Subheader
st.header("This is header")
st.subheader("This is a subheader")

# Text
st.text("Hello Streamlit")

# Markdown
st.markdown("### This is a Markdown")

# Error/Colorful Text
st.success("Successful")

st.info("Information")

st.warning("This is a warning")

st.error("This is an error! danger!")

st.exception("NameError('name three not defined')")

# Get help info about phyton
st.help(range)

# Writing ext
st.write("Text with write")

st.write(range(10))

# Images
from PIL import Image
img = Image.open("lenna.png")
st.image(img, width=300, caption="Simple Image")

# Videos
# vid_file = open("example.mp4", "rb")
# vid_byte = vid_file.read()
# st.video(vid_byte)

# Audio
# audio_file = open("examplemusic.mp3", "rb").read()
# st.audio(audio_file.format='audio/mp3')

# Widget
# Checkbox
if st.checkbox("Show/Hide"):
    st.text("Showing or Hiding Widget")

# Radio Buttons
status = st.radio("What is your status", ("Active", "Inactive"))

if status == 'Active':
    st.success("You are Active")
else:
    st.warning("Inactive, Activate")

# Selectbox
occupation = st.selectbox("Your Occupation", ["Programmer","DataScientist", "Doctor", "Businessman"])
st.write("You selected this option ", occupation)

# MultiSelect
location = st.multiselect("Where do you work?", ("London","New York", "Accra", "Kiev", "Nepal"))
st.write("You selected", len(location), "locations")

# Slider
level = st.slider("What is your level", 1, 5)

# Button
st.button("Simple Button")

if st.button("About"):
    st.text("Streamlit is Cool")


# Text Input
firstname = st.text_input("Enter Your Firstname", "Type Here..")
if st.button("Submit", key="1"):
    result = firstname.title()
    st.success(result)


# Text Area
message = st.text_area("Enter Your message", "Type Here..")
if st.button("Submit", key="2"):
    result = message.title()
    st.success(result)

# Date Input
import datetime
today = st.date_input("Today is",datetime.datetime.now())

# Time
the_time = st.time_input("The time is ",datetime.time())

# Display JSON
st.text("Display JSON")
st.json({'name':"Jesse", 'gender':"male"})

# Display Raw Code
st.text("Display Raw Code")
st.code("import numpy as np")

# Display Raw Code
with st.echo():
    # This is show as a comment
    import pandas as pd 
    df = pd.DataFrame()

# Progress Bar
import time
my_bar = st.progress(0)
for p in range(10):
    my_bar.progress(p + 1)

# Spinner
# with st.spinner("Waiting..."):
#     time.sleep(5)
# st.success("Finished!")

# Balloons
# st.balloons()

# Sidebars
st.sidebar.header("About")
st.sidebar.text("This is Streamlit Tutorials")

# Functions
@st.cache #increase performance
def run_fxn():
    return range(100)

st.write(run_fxn())

# Plot
st.pyplot()

# DataFrame
st.dataframe(df)

# Tables
st.table(df)