import streamlit as st

# To hide hamburger (top right corner) and “Made with Streamlit” footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Title
st.title('This is a title')

# Heading
st.header('This is a header')

# Subheading
st.subheader('This is a subheader')

# Text
st.text('This is text text text text')

# Markdown, less # implies larger font
st.markdown('# This is markdown')

# Ref. link
st.markdown('[Google link](https://google.com)')

url_link = 'https://google.com'
st.markdown(url_link)

# Custom HTML and Styling
html_page = """
                <div style="background-color:tomato;padding:50px>
                    <p style="font-size:50px">Custom HTML and Styling</p>
                </div>
            """
st.markdown(html_page, unsafe_allow_html=True)

# Form
html_form = """
            <div>
                <form>
                    <input type="text" name="firstname" />
                </form>
            </div>
            """
st.markdown(html_form, unsafe_allow_html=True)

# Bootstrap Alert/Color Text
st.success("Success!")
st.info("Information")
st.warning("Warning")
st.error("Error")
st.exception("NameError()")

# Images
from PIL import Image
img = Image.open("image1.jpg")
st.image(img, width=300, caption="Beautiful View")

# Audio
# audio_file = open("example.mp3, "rb")
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format="audio/mp3")

# Video
# video_file = open("example.mp4, "rb")
# video_bytes = video_file.read()
# st.video(video_bytes)

# Video from URL
st.video("https://www.youtube.com/watch?v=p0AejVoKT78")

# Button
st.button("Submit")

# Button with functionality
if st.button("Display Text"):
    st.text("Hello World!")

# Checkbox
st.checkbox("Select")
if st.checkbox("Show/hide"):
    st.success("Hiding or Showing")

# Radio button
state = st.radio("Liquid or Solid: ", ["Liquid", "Solid"])
if state=="Liquid":
    st.info("E.g. water")

# Dropdown menu single selection
location = st.selectbox("Your location", ["UK", "USA", "Germany", "Ireland"])

# Dropdown menu multi-selection
subjects = st.multiselect("Your favorite", ["Mathematics", "Physics", "Chemistry"])

# Text input
month = st.text_input("Current month", "Type here")
st.text(month)

# Bounded number input
date = st.number_input("Today's date", 1, 31)

# Text area
message = st.text_area("Your message", "Type here")

# Slider
rating = st.slider("Review rating", 1, 5) # Default limits (0, 100)

# Balloons
# st.balloons()

# Data science
st.write(range(10))

# Dataframe
import pandas as pd
df = pd.read_csv("iris.csv")
st.dataframe(df.head())
st.write(df.head())

# Tables
st.table(df.head())

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

df = df.drop(columns=['species'], axis=0)

st.area_chart(df.head(20))

st.bar_chart(df.head(20))

st.line_chart(df.head(20))

c_plot = sns.heatmap(df.corr(), annot=True)
st.write(c_plot)
st.pyplot()

# Data/Time
import datetime
today = st.date_input("The date today is", datetime.datetime.now())

import time
the_time = st.time_input("The time is", datetime.time(10,0))

# Display JSON
data = {"name": "John", "salary":50000}
st.json(data)

# Display code
st.code("import numpy as np")
st.code("import numpy as np", language='python')

with st.echo():
    # This is a comment
    import textblob

# Progressbar
import time
my_bar = st.progress(0)
for p in range(61):
    my_bar.progress(p+1)

# Spinner
with st.spinner("Spinner in progress..."):
    time.sleep(5)
st.success("Finished")