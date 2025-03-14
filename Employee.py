import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

# Setting up the page config
st.set_page_config(page_icon="🌐", page_title="Employee Attrition Analysis", layout="wide")

st.title(''':orange[***Welcome to the Employee Attrition Analysis system!***]''')

# Upload dataset through Streamlit (outside the page check)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
else:
    df = None  # If no file is uploaded, set df to None

# Set background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://getwallpapers.com/wallpaper/full/2/7/4/90076.jpg");
background-size: cover;
}

[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}   

[data-testid="stSidebarContent"]{
background-color: rgba(0, 0, 0, 0);
background-size: cover;
}
</style>
"""
# Display the background image
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar for page navigation
with st.sidebar:
    selected_page = option_menu('Employee Attrition Analysis and Prediction', 
                                ['Home', 'Age vs Employee-Attrition', 'Age Distribution', 'Department vs Employee-Attrition', 'Attrition Analysis', ],
                                icons=['house', 'bar-chart', 'person', 'briefcase', 'pie-chart'], 
                                default_index=0,
                                orientation='vertical',  
                                styles={
                                    "container": {"padding": "3px", "background-color": "#f0f0f5"},
                                    "icon": {"color": "orange", "font-size": "20px"},
                                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "3px", "--hover-color": "#f0f0f5"},
                                    "nav-link-selected": {"background-color": "#ADD8E6"},
                                }
                                )

# Home Page
if selected_page == 'Home':
    if df is None:
        st.write("Please upload a CSV file to proceed.")
    else:
        st.write("CSV file uploaded successfully!")

# Age vs Employee-Attrition Chart
elif selected_page == 'Age vs Employee-Attrition':
    if df is not None:
        st.title("Age vs Employee-Attrition Analysis")
        # Scatter plot of Age vs Employee-Attrition
        fig, ax = plt.subplots()
        ax.scatter(df['Age'], df['Attrition'], color='Slateblue', alpha=0.5)
        ax.set_xlabel('Age')
        ax.set_ylabel('Attrition')
        ax.set_title('Age vs Employee-Attrition')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to view this chart.")

# Age Distribution Chart
elif selected_page == 'Age Distribution':
    if df is not None:
        st.title("Age Distribution Analysis")
        # Histogram of Age
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution of Employees')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to view this chart.")

# Department vs Employee-Attrition (Attrition Analysis) Page
elif selected_page == 'Department vs Employee-Attrition':
    if df is not None:
        st.title("Department vs Employee-Attrition Analysis")
        # Boxplot to show Department vs Salary by Attrition
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Department', y='Attrition', hue='RelationshipSatisfaction', data=df, palette='Set2')
        plt.title("Department vs Employee-Attrition")
        plt.xlabel("Department")
        plt.ylabel("Attrition")
        st.pyplot(plt)
    else:   
        st.write("Please upload a CSV file to view this chart.")

# Attrition Analysis (Pie Chart)
elif selected_page == 'Attrition Analysis':
    if df is not None:
        st.title("Attrition Analysis")
        # Pie chart for Attrition distribution
        attrition_counts = df['Attrition'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(attrition_counts, labels=['No Attrition', 'Attrition'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
        ax.set_title('Attrition Distribution')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to view this chart.")