import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np

# To hide hamburger (top right corner) and “Made with Streamlit” footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

df = pd.read_csv("iris.csv")
st.dataframe(df)

df.plot(kind="bar")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Maps
df2 = pd.DataFrame(np.random.randn(500,2)/[50,50] + [37.76, -127.4], \
                        columns=["latitude", "longitude"])
st.map(df2)

# File selector
import os

# @st.cache() # For faster loading from cache
def file_selector(folder_path='./datasets'): # . for current folder, ./datasets for subfolder, directory etc.
    filenames = [file_name for file_name in os.listdir(folder_path) if file_name.endswith((".csv"))]
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected %s' % filename)
df = pd.read_csv(filename)
st.write(df)

# Sidebar (Navigation bar)
st.sidebar.header("About")
st.sidebar.text("Just a text")
st.sidebar.header("Analyze")

# CSS logo
def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

load_css('icon.css')
load_icon('face')
load_icon('code')