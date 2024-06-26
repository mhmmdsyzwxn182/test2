import streamlit as st



import pandas as pd

import home
import data
import plots
import predicts
import about

from load_data import load_data

st.set_page_config(
    page_title = 'Car Price Prediction',
    page_icon = 'car',
    layout = 'centered',
    initial_sidebar_state = 'auto'
)

# Create a dict for pages.
pages_dict = {
                "Home": home,
                "View Data": data, 
                "Visualise Data": plots, 
                "Predict": predicts,
                "About me": about
            }

# Load the dataset.
df = load_data()

# Create navbar in sidebar.
st.sidebar.title("Navigation")
user_choice = st.sidebar.radio('Go to', ("Home", "View Data", "Visualise Data", "Predict", "About me"))

# Open the page selected by the user.
if (user_choice == "Home" or user_choice == "About me"):
    selected_page = pages_dict[user_choice]
    selected_page.app()
else:
    selected_page = pages_dict[user_choice]
    selected_page.app(df)
